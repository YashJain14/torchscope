"""
Example: profile the video-inference-benchmark under torchscope.

This is ONE example of torchscope usage — the video benchmark is used
as a case study, not the core purpose of the tool.

    python examples/video_pipeline.py \
        --video  ../../ToS.mp4 \
        --model  ../../yolov8n.pt \
        --pipelines opencv_cpu ffmpeg_dali pynvvideocodec pynv_cvcuda

torchscope wraps the benchmark harness and enriches the report with:
  - Live GPU utilization timeline (shows idle gaps during CPU decode)
  - Memory waterfall
  - Stage-level bottleneck detection
  - Actionable recommendations
"""

import sys
import argparse
from pathlib import Path

# locate repo root so both torchscope and benchmark.py are importable
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE.parent))

from torchscope import Profiler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        required=True)
    ap.add_argument("--model",        required=True)
    ap.add_argument("--total-frames", type=int, default=512)
    ap.add_argument("--batch",        type=int, default=16)
    ap.add_argument("--warmup",       type=int, default=3)
    ap.add_argument("--runs",         type=int, default=10)
    ap.add_argument("--pipelines",    nargs="+",
                    default=["opencv_cpu", "ffmpeg_dali",
                             "pynvvideocodec", "pynv_cvcuda"])
    ap.add_argument("--report",       default="torchscope_video.html")
    ap.add_argument("--out-csv",      default="results/benchmark_results.csv")
    args = ap.parse_args()

    # import benchmark harness
    try:
        from benchmark import BenchmarkConfig, run_benchmark, save_csv
    except ImportError:
        print("benchmark.py not found on path. Run from the repo root or pass correct paths.")
        sys.exit(1)

    from integrations.benchmark_adapter import adapt_results

    config = BenchmarkConfig(
        video_path     = args.video,
        model_path     = args.model,
        total_frames   = args.total_frames,
        batch_size     = args.batch,
        warmup_runs    = args.warmup,
        benchmark_runs = args.runs,
    )

    prof = Profiler(
        interval    = 0.5,
        job_name    = "video_inference",
        export_json = "logs/video_pipeline.ndjson",
    )

    prof.start()
    results = run_benchmark(config, args.pipelines)
    prof.stop()

    save_csv(results, Path(args.out_csv))

    # convert benchmark stage timings into torchscope's custom_stages format
    custom_stages, pipeline_meta = adapt_results(results)

    prof.report(
        output_path   = args.report,
        title         = f"Video Inference Pipeline — {Path(args.video).name}",
        custom_stages = custom_stages,
        metadata      = {
            "video":        args.video,
            "batch_size":   args.batch,
            "total_frames": args.total_frames,
            **pipeline_meta,
        },
    )


if __name__ == "__main__":
    main()
