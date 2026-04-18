"""
Tests for NCCLCollector, _parse_nccl_line, and NCCL analyzer rules.
All tests run without a real GPU or torch.distributed initialization.
"""

import time
import pytest

from torchscope.collectors.nccl import (
    NCCLCollector, NCCLEvent, _parse_nccl_line, _estimate_size,
)
from torchscope.analyzer import BottleneckAnalyzer


# ── log parsing ───────────────────────────────────────────────────────────────

class TestNCCLLogParsing:
    def test_parse_allreduce_line(self):
        line = (
            "NCCL INFO AllReduce: sendbuff=0xABCD count=1048576 "
            "datatype=float32 op=Sum algorithm=Ring protocol=Simple"
        )
        ev = _parse_nccl_line(line)
        assert ev is not None
        assert ev.op == "all_reduce"
        assert ev.size_bytes == 1048576 * 4
        assert ev.algorithm == "ring"
        assert ev.protocol  == "SIMPLE"

    def test_parse_allgather_line(self):
        line = (
            "NCCL INFO AllGather: count=65536 datatype=float16 "
            "algorithm=Tree protocol=LL"
        )
        ev = _parse_nccl_line(line)
        assert ev is not None
        assert ev.op == "all_gather"
        assert ev.size_bytes == 65536 * 2
        assert ev.algorithm == "tree"
        assert ev.protocol  == "LL"

    def test_parse_broadcast_line(self):
        line = "NCCL INFO Broadcast: count=256 datatype=float64"
        ev = _parse_nccl_line(line)
        assert ev is not None
        assert ev.op == "broadcast"
        assert ev.size_bytes == 256 * 8

    def test_non_nccl_line_returns_none(self):
        assert _parse_nccl_line("random log line") is None
        assert _parse_nccl_line("") is None
        assert _parse_nccl_line("INFO something else happened") is None

    def test_unknown_dtype_defaults_to_4_bytes(self):
        line = "NCCL INFO AllReduce: count=100 datatype=custom_quant_type"
        ev = _parse_nccl_line(line)
        assert ev is not None
        assert ev.size_bytes == 100 * 4

    def test_bfloat16_size(self):
        line = "NCCL INFO AllReduce: count=512 datatype=bfloat16"
        ev = _parse_nccl_line(line)
        assert ev.size_bytes == 512 * 2

    def test_duration_ms_zero_in_log_mode(self):
        line = "NCCL INFO AllReduce: count=100 datatype=float32"
        ev = _parse_nccl_line(line)
        assert ev.duration_ms == 0.0


# ── collector summary ─────────────────────────────────────────────────────────

def _make_event(op, size_bytes, duration_ms):
    return NCCLEvent(op=op, size_bytes=size_bytes, duration_ms=duration_ms,
                     algorithm="", protocol="")


class TestNCCLCollectorSummary:
    def _col_with_events(self, events):
        col = NCCLCollector(mode="hook")
        col.events = events
        return col

    def test_empty_summary(self):
        col = NCCLCollector(mode="hook")
        s = col.summary()
        assert s["n_events"] == 0

    def test_dominant_collective_allreduce(self):
        col = self._col_with_events([
            _make_event("all_reduce", 4096, 80.0),
            _make_event("all_gather", 4096, 20.0),
        ])
        s = col.summary()
        assert s["dominant_collective"] == "all_reduce"
        assert s["dominant_pct"] == pytest.approx(80.0)

    def test_dominant_collective_allgather(self):
        col = self._col_with_events([
            _make_event("all_reduce", 4096, 10.0),
            _make_event("all_gather", 4096, 90.0),
        ])
        s = col.summary()
        assert s["dominant_collective"] == "all_gather"

    def test_avg_message_size(self):
        col = self._col_with_events([
            _make_event("all_reduce", 2 * 1024 * 1024, 50.0),
            _make_event("all_reduce", 4 * 1024 * 1024, 50.0),
        ])
        s = col.summary()
        assert s["avg_message_size_bytes"] == 3 * 1024 * 1024

    def test_events_by_op_counts(self):
        col = self._col_with_events([
            _make_event("all_reduce", 1024, 10.0),
            _make_event("all_reduce", 1024, 20.0),
            _make_event("all_gather", 2048, 15.0),
        ])
        s = col.summary()
        assert s["events_by_op"]["all_reduce"]["count"] == 2
        assert s["events_by_op"]["all_reduce"]["total_ms"] == pytest.approx(30.0)
        assert s["events_by_op"]["all_gather"]["count"] == 1

    def test_scalar_op_fields(self):
        col = self._col_with_events([
            _make_event("all_reduce",    1024, 40.0),
            _make_event("all_gather",    1024, 30.0),
            _make_event("reduce_scatter", 1024, 20.0),
        ])
        s = col.summary()
        assert s["allreduce_ms"]     == pytest.approx(40.0)
        assert s["allgather_ms"]     == pytest.approx(30.0)
        assert s["reducescatter_ms"] == pytest.approx(20.0)

    def test_total_ms(self):
        col = self._col_with_events([
            _make_event("all_reduce", 1024, 50.0),
            _make_event("broadcast",  1024, 50.0),
        ])
        s = col.summary()
        assert s["total_ms"] == pytest.approx(100.0)


# ── analyzer NCCL rules ───────────────────────────────────────────────────────

class TestAnalyzerNCCLRules:
    def _analyze(self, **kwargs):
        return BottleneckAnalyzer(**kwargs).analyze()

    def test_allreduce_dominant_fires(self):
        result = self._analyze(nccl={
            "dominant_collective": "all_reduce",
            "dominant_pct":        75.0,
            "allreduce_ms":        450.0,
            "n_events":            100,
        })
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_ALLREDUCE_DOMINANT" in types

    def test_allreduce_dominant_below_threshold(self):
        result = self._analyze(nccl={
            "dominant_collective": "all_reduce",
            "dominant_pct":        50.0,   # below 60%
            "n_events":            100,
        })
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_ALLREDUCE_DOMINANT" not in types

    def test_allreduce_dominant_not_allreduce(self):
        result = self._analyze(nccl={
            "dominant_collective": "all_gather",
            "dominant_pct":        80.0,
            "n_events":            100,
        })
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_ALLREDUCE_DOMINANT" not in types

    def test_small_message_fires(self):
        result = self._analyze(nccl={
            "avg_message_size_bytes": 65536,   # 64 KB < 1 MB
            "n_events":               50,
        })
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_SMALL_MESSAGE" in types

    def test_small_message_above_threshold(self):
        result = self._analyze(nccl={
            "avg_message_size_bytes": 10_000_000,  # 10 MB
            "n_events":               50,
        })
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_SMALL_MESSAGE" not in types

    def test_small_message_no_events(self):
        result = self._analyze(nccl={"avg_message_size_bytes": 100, "n_events": 0})
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_SMALL_MESSAGE" not in types

    def test_empty_nccl_no_false_positives(self):
        result = self._analyze(nccl={})
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_ALLREDUCE_DOMINANT" not in types
        assert "NCCL_SMALL_MESSAGE"      not in types

    def test_none_nccl_no_false_positives(self):
        result = self._analyze()
        types = [f["type"] for f in result["findings"]]
        assert "NCCL_ALLREDUCE_DOMINANT" not in types
        assert "NCCL_SMALL_MESSAGE"      not in types

    def test_allreduce_finding_severity_medium(self):
        result = self._analyze(nccl={
            "dominant_collective": "all_reduce",
            "dominant_pct":        80.0,
            "allreduce_ms":        500.0,
            "n_events":            50,
        })
        f = next(x for x in result["findings"] if x["type"] == "NCCL_ALLREDUCE_DOMINANT")
        assert f["severity"] == "MEDIUM"

    def test_small_message_finding_severity_medium(self):
        result = self._analyze(nccl={"avg_message_size_bytes": 1024, "n_events": 10})
        f = next(x for x in result["findings"] if x["type"] == "NCCL_SMALL_MESSAGE")
        assert f["severity"] == "MEDIUM"

    def test_recommendations_populated(self):
        result = self._analyze(nccl={
            "dominant_collective": "all_reduce",
            "dominant_pct":        70.0,
            "allreduce_ms":        200.0,
            "n_events":            20,
        })
        assert len(result["recommendations"]) >= 1
        assert any("bucket_cap_mb" in r for r in result["recommendations"])


# ── lifecycle (no dist) ───────────────────────────────────────────────────────

class TestNCCLHookLifecycle:
    def test_start_stop_no_distributed(self):
        col = NCCLCollector(mode="hook")
        col.start()
        col.stop()
        assert len(col.events) == 0

    def test_double_stop_no_raise(self):
        col = NCCLCollector(mode="hook")
        col.start()
        col.stop()
        col.stop()

    def test_log_mode_missing_file(self):
        col = NCCLCollector(mode="log", log_path="/nonexistent/path/nccl.log")
        col.start()
        time.sleep(0.1)
        col.stop()
        assert len(col.events) == 0

    def test_log_mode_no_path(self):
        col = NCCLCollector(mode="log", log_path=None)
        col.start()
        time.sleep(0.1)
        col.stop()
        assert len(col.events) == 0

    def test_estimate_size_no_torch_args(self):
        assert _estimate_size(()) == 0
        assert _estimate_size(("not_a_tensor",)) == 0
