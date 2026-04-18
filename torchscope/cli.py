"""
torchscope CLI

Commands:
    torchscope run      -- wrap a Python script with live profiling
    torchscope analyze  -- generate a report from existing CSV / NDJSON
    torchscope info     -- print environment info (GPU, torch, pynvml)

Examples:
    torchscope run train.py --args "--epochs 3 --batch 64" --report out.html
    torchscope analyze results/benchmark.csv --report out.html
    torchscope info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── sub-commands ──────────────────────────────────────────────────────────────

def cmd_run(args):
    """
    Run a Python script under torchscope.

    Injects profiler.start() before the script and profiler.report() after
    by running it in a subprocess with PYTHONSTARTUP injection.
    """
    import os, subprocess, tempfile, textwrap

    script = Path(args.script)
    if not script.exists():
        print(f"torchscope: script not found: {script}", file=sys.stderr)
        sys.exit(1)

    # Build a small wrapper that imports torchscope, starts profiling,
    # exec's the target script, then writes the report.
    wrapper = textwrap.dedent(f"""
        import sys, runpy
        sys.path.insert(0, {str(script.parent)!r})

        # parse extra args into sys.argv before the script sees them
        extra = {args.script_args!r}
        if extra:
            sys.argv = [sys.argv[0]] + extra.split()

        import torchscope
        _prof = torchscope.Profiler(
            interval    = {args.interval},
            job_name    = {script.stem!r},
            export_json = {(args.json or '')!r} or None,
        )
        _prof.start()
        try:
            runpy.run_path({str(script)!r}, run_name="__main__")
        finally:
            _prof.report(
                output_path = {args.report!r},
                title       = {script.stem!r},
            )
    """).strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, prefix="torchscope_") as f:
        f.write(wrapper)
        tmp = f.name

    try:
        env = os.environ.copy()
        ret = subprocess.run([sys.executable, tmp], env=env)
        sys.exit(ret.returncode)
    finally:
        Path(tmp).unlink(missing_ok=True)


def cmd_analyze(args):
    """Generate a report from one or more benchmark CSVs or NDJSON files."""
    import csv as _csv
    from collections import defaultdict

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from torchscope.analyzer       import BottleneckAnalyzer
    from torchscope.exporters.html import HTMLExporter

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            print(f"torchscope: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # load CSVs
    rows = []
    for p in paths:
        if p.suffix == ".csv":
            rows.extend(_load_csv(p))
        elif p.suffix in (".ndjson", ".jsonl"):
            rows.extend(_load_ndjson(p))
        else:
            print(f"torchscope: unsupported file type: {p.suffix}", file=sys.stderr)
            sys.exit(1)

    if not rows:
        print("torchscope: no valid rows found in input files.", file=sys.stderr)
        sys.exit(1)

    custom_stages = _stages_from_rows(rows)

    analyzer = BottleneckAnalyzer(custom_stages=custom_stages)
    analysis = analyzer.analyze()
    analysis["_gpu_summary"] = {
        "avg_gpu_util": "N/A (offline)",
        "peak_mem_gb":  "N/A (offline)",
        "avg_power_w":  "N/A (offline)",
        "duration_s":   "N/A (offline)",
    }

    HTMLExporter().generate(
        output_path   = args.report,
        title         = args.title,
        analysis      = analysis,
        custom_stages = custom_stages,
        metadata      = {"source files": ", ".join(str(p) for p in paths)},
    )

    # print findings to stdout
    print(f"\ntorchscope analyze — {len(rows)} rows from {len(paths)} file(s)")
    for f in analysis["findings"]:
        icon = {"HIGH": "●", "MEDIUM": "◐", "LOW": "○"}.get(f["severity"], "·")
        print(f"  {icon} [{f['severity']}] {f['type']}: {f['detail']}")
    for r in analysis["recommendations"]:
        print(f"    → {r}")
    print(f"\nReport: {args.report}")


def cmd_info(_args):
    """Print environment diagnostics."""
    import platform
    print(f"torchscope info")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Platform : {platform.system()} {platform.machine()}")

    try:
        import torch
        print(f"  PyTorch  : {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem   = props.total_memory / 1e9
                print(f"  GPU {i}     : {props.name}  {mem:.1f} GB")
        else:
            print("  GPU      : not available")
    except ImportError:
        print("  PyTorch  : not installed")

    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        print(f"  pynvml   : OK ({n} device(s))")
    except Exception as e:
        print(f"  pynvml   : {e}")

    try:
        import plotly
        print(f"  plotly   : {plotly.__version__}")
    except ImportError:
        print("  plotly   : not installed")

    try:
        import jinja2
        print(f"  jinja2   : {jinja2.__version__}")
    except ImportError:
        print("  jinja2   : not installed")

    try:
        import prometheus_client
        print(f"  prometheus-client : {prometheus_client.__version__}")
    except ImportError:
        print("  prometheus-client : not installed  (pip install prometheus-client)")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict]:
    import csv
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            for k in ("decode_s", "preprocess_s", "inference_s", "postprocess_s",
                      "total_s", "fps", "frames_processed", "run_id"):
                try:    row[k] = float(row[k])
                except: row[k] = 0.0
            rows.append(row)
    return rows


def _load_ndjson(path: Path) -> list[dict]:
    import json
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("record_type") == "run_summary":
                    rows.append(obj)
            except json.JSONDecodeError:
                pass
    return rows


def _stages_from_rows(rows: list[dict]) -> dict:
    """Best-effort: average stage timings from CSV rows."""
    stage_keys = ["decode_s", "preprocess_s", "inference_s", "postprocess_s"]
    totals = {k: 0.0 for k in stage_keys}
    n = 0
    for row in rows:
        if any(k in row for k in stage_keys):
            for k in stage_keys:
                totals[k] += float(row.get(k, 0))
            n += 1
    if n == 0:
        return {}
    return {k.replace("_s", ""): totals[k] / n for k in stage_keys if totals[k] > 0}


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="torchscope",
        description="PyTorch GPU observability toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # run
    p_run = sub.add_parser("run", help="Profile a Python script")
    p_run.add_argument("script",                    help="Script to run")
    p_run.add_argument("--args",   dest="script_args", default="",
                       help="Arguments to pass to the script (quoted string)")
    p_run.add_argument("--report", default="torchscope_report.html")
    p_run.add_argument("--json",   default=None,    help="NDJSON log output path")
    p_run.add_argument("--interval", type=float, default=0.5)
    p_run.set_defaults(func=cmd_run)

    # analyze
    p_ana = sub.add_parser("analyze", help="Report from existing CSV / NDJSON files")
    p_ana.add_argument("inputs", nargs="+",         help="CSV or NDJSON files")
    p_ana.add_argument("--report", default="torchscope_report.html")
    p_ana.add_argument("--title",  default="torchscope analysis")
    p_ana.set_defaults(func=cmd_analyze)

    # info
    p_inf = sub.add_parser("info", help="Print environment diagnostics")
    p_inf.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
