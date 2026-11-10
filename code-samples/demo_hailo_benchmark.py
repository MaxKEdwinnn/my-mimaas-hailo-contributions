"""Run a quick, end-to-end MIMaaS benchmark on the Hailo-10H Pi 5.

The demo uses an existing HEF, so no Hailo DFC compilation is needed.
Run from the repository root:
    PYTHONPATH=src python scripts/demo_hailo_benchmark.py \
        --model hailo_models/classification/hardnet39ds.hef \
        --output data/results/hardnet39ds_benchmark.json
"""
import argparse
import json
from pathlib import Path

from mimaas.core.pipeline import flash_and_evaluate
from mimaas.models.board import Board

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "hailo_models" / "classification" / "hardnet39ds.hef"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Model Zoo HEF through the MIMaaS Hailo benchmark pipeline."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to an existing .hef file (default: hardnet39ds Model Zoo HEF).",
    )
    parser.add_argument(
        "--board",
        default="hailo_pi5",
        help="Board configuration name from config/boards (default: hailo_pi5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the combined MIMaaS and HailoRT JSON report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.resolve()
    if model_path.suffix.lower() != ".hef":
        raise SystemExit(f"Expected a .hef file, received: {model_path}")
    if not model_path.is_file():
        raise SystemExit(f"HEF file not found: {model_path}")

    board = Board.from_config(args.board)

    print(f"Board: {board.name} ({board.host})")
    print(f"Model: {model_path}")
    print("Running MIMaaS Hailo benchmark...")

    result = flash_and_evaluate(str(model_path), board)
    print("Pipeline result:")
    print(json.dumps(result, indent=2))

    metrics_path = model_path.parent / "hailo_results.json"
    if result.get("status") != "flash_and_eva_successful" or not metrics_path.exists():
        raise SystemExit("Benchmark did not complete successfully; see the pipeline output above.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = {
        "model": str(model_path),
        "board": args.board,
        "mimaas": result,
        "hailort": metrics,
    }
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report exported to: {output_path}")

    print("\nHardware measurements:")
    print(f"  FPS: {metrics.get('fps', 0):.2f}")
    print(f"  Throughput-derived latency: {metrics.get('inference_time_ms', 0):.3f} ms")
    print(f"  HEF ROM size: {metrics.get('rom_usage_mb', 0):.3f} MB")
    if metrics.get("temp_mean_c") is not None:
        print(
            "  Hailo-10H temperature: "
            f"mean={metrics['temp_mean_c']:.2f} C, "
            f"min={metrics.get('temp_min_c', 0):.2f} C, "
            f"max={metrics.get('temp_max_c', 0):.2f} C"
        )


if __name__ == "__main__":
    main()
