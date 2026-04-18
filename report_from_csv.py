"""
Offline report from existing benchmark CSVs — no GPU required.

    python report_from_csv.py --csv results/full_sweep_16.csv --report out.html
    python report_from_csv.py \
        --csv results/full_sweep_16.csv results/full_sweep_32.csv \
               results/full_sweep_64.csv results/full_sweep_128.csv \
        --report torchscope_report.html
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from torchscope.analyzer        import BottleneckAnalyzer
from torchscope.exporters.html  import HTMLExporter
from integrations.benchmark_adapter import adapt_results


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            for k in ("decode_s","preprocess_s","inference_s","postprocess_s",
                      "total_s","fps","frames_processed","run_id"):
                try:    row[k] = float(row[k])
                except: row[k] = 0.0
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    nargs="+", required=True)
    ap.add_argument("--report", default="torchscope_report.html")
    ap.add_argument("--title",  default="Video Inference Pipeline")
    args = ap.parse_args()

    rows = []
    for p in args.csv:
        rows.extend(load_csv(Path(p)))

    if not rows:
        print("No valid rows found.")
        sys.exit(1)

    custom_stages, pipeline_meta = adapt_results(rows)

    analyzer = BottleneckAnalyzer(
        gpu           = {},
        memory        = {},
        comm          = {},
        custom_stages = custom_stages,
    )
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
        metadata      = {"source CSVs": ", ".join(args.csv), **pipeline_meta},
    )

    print("\nFindings:")
    for f in analysis["findings"]:
        print(f"  [{f['severity']}] {f['type']}: {f['detail']}")
    if analysis["recommendations"]:
        print("\nRecommendations:")
        for r in analysis["recommendations"]:
            print(f"  → {r}")


if __name__ == "__main__":
    main()
