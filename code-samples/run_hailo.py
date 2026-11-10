#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - environment dependent fallback
    psutil = None


def _estimate_ram_usage_mb(path: Path) -> float:
    try:
        total_bytes = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total_bytes += entry.stat().st_size
        return round(total_bytes / (1024 * 1024), 3)
    except OSError:
        return 0.0


def _get_host_ram_usage_mb(path: Path) -> float:
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / (1024 * 1024), 3)
        except Exception:
            pass
    return _estimate_ram_usage_mb(path)


def _flatten_dict_items(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            key_name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_dict_items(item, key_name)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            key_name = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _flatten_dict_items(item, key_name)
    else:
        yield prefix, value


def _find_first_numeric(flat_items, aliases):
    aliases_lower = [alias.lower() for alias in aliases]
    for key, value in flat_items:
        key_lower = key.lower()
        if any(alias in key_lower for alias in aliases_lower) and isinstance(value, (int, float)):
            return float(value)
    return None


def _find_first_text(flat_items, aliases):
    aliases_lower = [alias.lower() for alias in aliases]
    for key, value in flat_items:
        key_lower = key.lower()
        if any(alias in key_lower for alias in aliases_lower) and isinstance(value, str):
            return value
    return None


def _parse_fps_from_hailortcli(output: str):
    """Extract FPS from hailortcli benchmark summary; returns None if not found."""
    match = re.search(r"FPS:\s*([\d.]+)", output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _parse_temperature_from_hailortcli(output: str):
    """Extract mean/min/max chip temperature (°C) from hailortcli output."""
    match = re.search(
        r"temperature:\s*mean=([\d.]+)\s*min=([\d.]+)\s*max=([\d.]+)", output
    )
    if not match:
        return None, None, None
    try:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    except ValueError:
        return None, None, None


def _parse_dfc_report(dfc_report_path: Path):
    if not dfc_report_path.exists():
        return None, "missing"
    if dfc_report_path.suffix.lower() != ".json":
        return None, "unsupported_format"

    try:
        # PowerShell can emit UTF-8 with BOM; utf-8-sig handles both BOM and non-BOM inputs.
        payload = json.loads(dfc_report_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return None, "parse_error"

    flat_items = list(_flatten_dict_items(payload))
    dram_mb = _find_first_numeric(
        flat_items,
        ["npu_dram_mb", "dram_mb", "dram", "memory_dram", "weights_dram"],
    )
    sram_mb = _find_first_numeric(
        flat_items,
        ["npu_sram_mb", "sram_mb", "sram", "memory_sram", "weights_sram"],
    )
    compiler_version = _find_first_text(flat_items, ["compiler_version", "dfc_version", "hailo_dfc"])

    report = {
        "dram_mb": dram_mb,
        "sram_mb": sram_mb,
        "compiler_version": compiler_version,
    }
    return report, "dfc_json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a compiled HEF model with HailoRT")
    parser.add_argument("--model", required=True, help="Path to the model file to run")
    parser.add_argument("--output", required=True, help="Path to the JSON output file")
    parser.add_argument("--dfc-report", required=False, help="Optional path to DFC JSON report for NPU memory metadata")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    result = {
        "status": "error",
        "model": str(model_path),
        "inference_time_ms": None,
        "ram_usage_mb": 0.0,
        "host_ram_usage_mb": 0.0,
        "rom_usage_mb": 0.0,
        "npu_memory_report": None,
        "npu_memory_report_source": "not_provided",
        "power_uw": None,
        "energy_uj": None,
        "measurement_status": "not_measured",
        "measurement_source": "unknown",
        "mode": "error",
        "note": "No real HailoRT inference was executed.",
    }

    if args.dfc_report:
        dfc_report, dfc_source = _parse_dfc_report(Path(args.dfc_report))
        result["npu_memory_report"] = dfc_report
        result["npu_memory_report_source"] = dfc_source

    if model_path.suffix.lower() != ".hef":
        result["note"] = f"Unsupported model format: {model_path.suffix}. A compiled .hef is required."
    else:
        try:
            proc = subprocess.run(["hailortcli", "benchmark", str(model_path)], capture_output=True, text=True, check=True)
            result["status"] = "success"
            result["mode"] = "hailortcli"
            result["note"] = proc.stdout.strip() or "Executed via hailortcli benchmark"
            result["measurement_status"] = "not_measured"
            result["measurement_source"] = "unknown"
            result["rom_usage_mb"] = round(model_path.stat().st_size / (1024 * 1024), 3)
            fps = _parse_fps_from_hailortcli(proc.stdout)
            result["fps"] = fps
            if fps and fps > 0:
                result["inference_time_ms"] = round(1000.0 / fps, 3)
            temp_mean, temp_min, temp_max = _parse_temperature_from_hailortcli(proc.stdout)
            result["temp_mean_c"] = temp_mean
            result["temp_min_c"] = temp_min
            result["temp_max_c"] = temp_max
        except FileNotFoundError:
            result["note"] = "hailortcli not found; real HailoRT inference could not be executed"
        except subprocess.CalledProcessError as exc:
            result["note"] = (exc.stderr or exc.stdout or "hailortcli execution failed").strip()

    host_ram_usage_mb = _get_host_ram_usage_mb(output_path.parent)
    result["host_ram_usage_mb"] = host_ram_usage_mb
    result["ram_usage_mb"] = host_ram_usage_mb

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
