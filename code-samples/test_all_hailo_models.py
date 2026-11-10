#!/usr/bin/env python3
"""
Test all Hailo models against the Pi5 via the MIMaaS pipeline.
Generates a comprehensive test report with timing and measurements.
"""
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mimaas.core.pipeline import flash_and_evaluate
from mimaas.models.board import Board


def get_board() -> Board:
    """Create and return the Hailo Pi5 board config."""
    hailo_host = os.getenv("HAILO_HOST", "192.168.42.1")
    hailo_ssh_user = os.getenv("HAILO_SSH_USER", "maxime")

    return Board(
        name="hailo_pi5_1",
        variant="hailo_rpi5",
        board_snr="HAILO10_PI5_01",
        ppk2_snr="N/A",
        uart_port_type="none",
        voltage=5000,
        power_measurement_threshold=1000,
        max_available_tensor_arena_size=0,
        flash_size=32768,
        ram_size=2048,
        board_type="hailo_rpi5",
        host=hailo_host,
        ssh_user=hailo_ssh_user,
        processor="Cortex-A76",
        npu="Hailo-10",
        supported_formats=[".hef", ".tflite"],
        max_model_size_mb=50,
    )


def discover_models(base_dir: Path) -> dict:
    """Discover all HEF models organized by category."""
    models = {}
    for category_dir in sorted(base_dir.iterdir()):
        if category_dir.is_dir():
            category = category_dir.name
            models[category] = []
            for hef_file in sorted(category_dir.glob("*.hef")):
                models[category].append(hef_file)
    return models


def test_model(model_path: Path, board: Board) -> dict:
    """Test a single model and return results."""
    logger.info(f"Testing: {model_path.name}")
    try:
        result = flash_and_evaluate(str(model_path), board)
        metrics = {}
        results_path = model_path.parent / "results.json"
        if results_path.exists():
            try:
                metrics = json.loads(results_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                metrics = {}

        return {
            "model": model_path.name,
            "path": str(model_path),
            "status": result.get("status", "unknown"),
            "duration_avg_s": metrics.get("duration_avg_s"),
            "inference_time_ms": metrics.get("duration_avg_s") * 1000 if metrics.get("duration_avg_s") is not None else None,
            "ram_usage_mb": metrics.get("ram_usage"),
            "host_ram_usage_mb": metrics.get("host_ram_usage_mb"),
            "rom_usage_mb": metrics.get("rom_usage"),
            "power_uw": metrics.get("avg_power_uW"),
            "energy_uj": metrics.get("avg_energy_uJ"),
            "measurement_status": metrics.get("measurement_status", "unknown"),
            "npu_memory_report": metrics.get("npu_memory_report"),
            "npu_memory_report_source": metrics.get("npu_memory_report_source"),
            "error": None,
        }
    except Exception as e:
        logger.error(f"Failed to test {model_path.name}: {e}")
        return {
            "model": model_path.name,
            "path": str(model_path),
            "status": "error",
            "error": str(e),
        }


def generate_report(results: dict) -> str:
    """Generate a markdown report from test results."""
    report_lines = [
        "# Hailo Model Test Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
    ]

    total_tests = sum(len(models) for models in results.values())
    successful = sum(1 for r in sum(results.values(), []) if r.get("status") == "flash_and_eva_successful")
    failed = sum(1 for r in sum(results.values(), []) if r.get("status") in ("error", "exception"))

    report_lines.append(f"- Total models tested: {total_tests}")
    report_lines.append(f"- Successful: {successful}")
    report_lines.append(f"- Failed: {failed}")
    report_lines.append("")

    for category, category_results in results.items():
        report_lines.append(f"## {category.replace('_', ' ').title()}")
        report_lines.append("")
        report_lines.append("| Model | Status | Inference (ms) | ROM (MB) | RAM (MB) |")
        report_lines.append("|-------|--------|-----------------|----------|----------|")

        for result in category_results:
            model_name = result.get("model", "unknown")
            status = result.get("status", "unknown")
            inference_ms = result.get("inference_time_ms", "-")
            rom_mb = result.get("rom_usage_mb", "-")
            ram_mb = result.get("ram_usage_mb", "-")

            if isinstance(inference_ms, (int, float)):
                inference_ms = f"{inference_ms:.2f}"
            if isinstance(rom_mb, (int, float)):
                rom_mb = f"{rom_mb:.2f}"
            if isinstance(ram_mb, (int, float)):
                ram_mb = f"{ram_mb:.2f}"

            report_lines.append(
                f"| {model_name} | {status} | {inference_ms} | {rom_mb} | {ram_mb} |"
            )

        report_lines.append("")

    report_lines.append("## Notes")
    report_lines.append("- Power and Energy marked as 'not_measured' (requires external probe)")
    report_lines.append("- Inference time measured from remote SSH execution elapsed time")
    report_lines.append("- ROM size = model file size (MB)")
    report_lines.append("- RAM column = legacy ram_usage for compatibility")
    report_lines.append("- Host RAM is also captured in JSON as host_ram_usage_mb")
    report_lines.append("- NPU memory report is captured in JSON when a DFC report is available")
    report_lines.append("")

    return "\n".join(report_lines)


def main():
    """Main test runner."""
    logger.info("=== Hailo Model Test Suite ===")

    base_dir = Path(__file__).parent.parent / "hailo_models"
    if not base_dir.exists():
        logger.error(f"Models directory not found: {base_dir}")
        return 1

    logger.info(f"Discovering models in: {base_dir}")
    models = discover_models(base_dir)

    if not models or all(not v for v in models.values()):
        logger.error("No HEF models found")
        return 1

    logger.info(f"Found {sum(len(v) for v in models.values())} models across {len(models)} categories")

    board = get_board()
    logger.info(f"Board target: {board.name} @ {board.host}")

    # Run tests
    results = {}
    for category, model_paths in models.items():
        logger.info(f"\nTesting {category} models...")
        results[category] = []
        for model_path in model_paths:
            result = test_model(model_path, board)
            results[category].append(result)

    # Generate report
    report = generate_report(results)

    # Save report
    report_path = Path(__file__).parent.parent / "HAILO_TEST_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {report_path}")

    # Save JSON results
    json_path = Path(__file__).parent.parent / "hailo_test_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Detailed results saved to: {json_path}")

    # Print summary
    print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
