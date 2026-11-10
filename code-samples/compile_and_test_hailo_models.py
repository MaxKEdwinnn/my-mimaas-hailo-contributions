#!/usr/bin/env python3
"""
Compile TFLite models to HEF and test them on MIMaaS.
Switch-based pipeline: if TFLite -> compile to HEF, if HEF -> test directly.
"""

import os
import json
import re
import subprocess
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
REPO_ROOT = Path(__file__).parent.parent
FILES_DIR = REPO_ROOT / "files"
HAILO_MODELS_DIR = REPO_ROOT / "hailo_models"
COMPILED_DIR = HAILO_MODELS_DIR / "compiled"
CUSTOM_DIR = HAILO_MODELS_DIR / "custom"
REPORT_FILE = REPO_ROOT / "HAILO_COMPILE_TEST_REPORT.md"
RESULTS_JSON = REPO_ROOT / "hailo_compile_test_results.json"

# DFC configuration (WSL)
DFC_VENV_NAME = "hailo_dfc_530_py310"
DFC_WORKDIR = "~/mimaas_dfc_test"
HW_ARCH = "hailo10h"
# Hailo-10h specs: 9MB × 5 clusters on-chip SRAM for weights
HAILO10H_TOTAL_SRAM_MB = 45.0
DFC_VERSION = "5.3.0"

# MIMaaS board config
HAILO_BOARD = {
    "name": "hailo_pi5_1",
    "variant": "hailo_rpi5",
    "board_snr": "HAILO10_PI5_01",
    "ppk2_snr": "N/A",
    "uart_port_type": "none",
    "voltage": 5000,
    "power_measurement_threshold": 1000,
    "max_available_tensor_arena_size": 0,
    "flash_size": 32768,
    "ram_size": 2048,
    "board_type": "hailo_rpi5",
    "host": "192.168.42.1",
    "ssh_user": "maxime",
    "processor": "Cortex-A76",
    "npu": "Hailo-10",
    "supported_formats": [".hef", ".tflite"],
    "max_model_size_mb": 50
}


def _parse_compiler_memory_stats(compiler_log: str) -> dict:
    """
    Parse memory utilization % from DFC compiler table output.
    Converts to absolute MB using Hailo-10h SRAM spec (45MB total).
    DRAM = 0 when model fits in on-chip SRAM (util < 100%).
    """
    # Matches last column (Memory Util) in the Total row of the compiler table
    match = re.search(r'\|\s*Total\s*\|[^|]+\|[^|]+\|\s*([\d.]+)%', compiler_log)
    if not match:
        return {}
    mem_util_pct = float(match.group(1))
    sram_mb = round(HAILO10H_TOTAL_SRAM_MB * mem_util_pct / 100.0, 3)
    # If utilization exceeds 100%, weights spill to DRAM
    dram_mb = round(max(0.0, sram_mb - HAILO10H_TOTAL_SRAM_MB), 3)
    sram_mb = min(sram_mb, HAILO10H_TOTAL_SRAM_MB)
    return {
        "npu_sram_mb": sram_mb,
        "npu_dram_mb": dram_mb,
        "compiler_version": DFC_VERSION,
        "hw_arch": HW_ARCH,
        "memory_util_pct": mem_util_pct,
        "total_sram_spec_mb": HAILO10H_TOTAL_SRAM_MB,
    }


def convert_path_to_wsl(windows_path: str) -> str:
    """Convert Windows path to WSL path."""
    # C:\Users\... -> /mnt/c/Users/...
    path = Path(windows_path)
    parts = path.parts
    if len(parts) > 0 and len(parts[0]) == 2 and parts[0][1] == ':':
        drive = parts[0][0].lower()
        rest = '/'.join(parts[1:])
        return f"/mnt/{drive}/{rest}"
    return str(windows_path)


def run_wsl_command(cmd: str, cwd: str = DFC_WORKDIR) -> Tuple[int, str, str]:
    """
    Execute a command in WSL with DFC environment activated.
    Returns: (exit_code, stdout, stderr)
    """
    # Reset to clean Linux PATH first — Windows PATH (with parentheses) breaks eval/pyenv init.
    # Activate the pyenv virtualenv directly via its activate script instead of pyenv init.
    full_cmd = f"""
unset PYTHONPATH
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"
source "$HOME/.pyenv/versions/{DFC_VENV_NAME}/bin/activate"
cd {cwd}
{cmd}
"""
    
    try:
        result = subprocess.run(
            ["wsl", "bash", "-c", full_cmd],
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout per command
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out (600s)"
    except Exception as e:
        return -1, "", str(e)


def compile_tflite_to_hef(tflite_path: str) -> Optional[str]:
    """
    Compile TFLite to HEF using DFC in WSL.
    Returns path to compiled HEF or None on failure.
    """
    tflite_abs = str(Path(tflite_path).resolve())
    model_name = Path(tflite_path).stem
    
    logger.info(f"Compiling TFLite: {model_name}")
    
    # Convert to WSL path
    tflite_wsl = convert_path_to_wsl(tflite_abs)
    
    # Step 1: Parser
    logger.info(f"  → Parser: TFLite → HAR")
    parser_cmd = (
        f'hailo parser tf "{tflite_wsl}" '
        f'--hw-arch {HW_ARCH} '
        f'--net-name {model_name} '
        f'--har-path {model_name}.har'
    )
    exit_code, stdout, stderr = run_wsl_command(parser_cmd)
    if exit_code != 0:
        logger.error(f"  ✗ Parser failed: {stderr}")
        return None
    logger.info(f"  ✓ HAR generated: {model_name}.har")
    
    # Step 2: Optimize
    logger.info(f"  → Optimize: HAR with random calib set")
    opt_cmd = (
        f'hailo optimize {model_name}.har '
        f'--hw-arch {HW_ARCH} '
        f'--use-random-calib-set '
        f'--output-har-path {model_name}_opt.har'
    )
    exit_code, stdout, stderr = run_wsl_command(opt_cmd)
    if exit_code != 0:
        logger.error(f"  ✗ Optimize failed: {stderr}")
        return None
    logger.info(f"  ✓ Optimized HAR: {model_name}_opt.har")
    
    # Step 3: Compiler
    logger.info(f"  → Compiler: HAR → HEF")
    compiler_cmd = (
        f'hailo compiler {model_name}_opt.har '
        f'--hw-arch {HW_ARCH} '
        f'--output-dir .'
    )
    exit_code, stdout, stderr = run_wsl_command(compiler_cmd)
    if exit_code != 0:
        logger.error(f"  ✗ Compiler failed: {stderr}")
        return None
    logger.info(f"  ✓ HEF compiled: {model_name}.hef")

    # Step 4: Generate DFC memory report from compiler logs
    # DFC logs (including the utilization table) go to stderr
    dfc_stats = _parse_compiler_memory_stats(stderr + "\n" + stdout)
    wsl_dfc_windows = f"\\\\wsl$\\Ubuntu\\home\\maxime\\mimaas_dfc_test\\{model_name}_dfc530.dfc.json"
    repo_dfc_path = COMPILED_DIR / f"{model_name}_dfc530.dfc.json"
    if dfc_stats:
        try:
            with open(wsl_dfc_windows, "w", encoding="utf-8") as f:
                json.dump(dfc_stats, f, indent=2)
            logger.info(
                f"  ✓ DFC report: npu_sram_mb={dfc_stats['npu_sram_mb']}MB "
                f"({dfc_stats['memory_util_pct']}% of {HAILO10H_TOTAL_SRAM_MB}MB), "
                f"npu_dram_mb={dfc_stats['npu_dram_mb']}MB"
            )
        except Exception as e:
            logger.warning(f"  ⚠ Could not write DFC report: {e}")
    else:
        logger.warning(f"  ⚠ Could not parse memory utilization from compiler output")

    # Step 5: Copy HEF (and DFC sidecar if present) to Windows repo
    repo_hef_path = COMPILED_DIR / f"{model_name}_dfc530.hef"
    logger.info(f"  → Copying HEF to repo: {repo_hef_path}")
    try:
        wsl_to_windows = f"\\\\wsl$\\Ubuntu\\home\\maxime\\mimaas_dfc_test\\{model_name}.hef"
        repo_hef_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(wsl_to_windows, repo_hef_path)
        logger.info(f"  ✓ HEF copied to: {repo_hef_path}")
        # Copy DFC sidecar alongside HEF so find_local_dfc_report() picks it up automatically
        if Path(wsl_dfc_windows).exists():
            shutil.copy(wsl_dfc_windows, repo_dfc_path)
            logger.info(f"  ✓ DFC sidecar copied: {repo_dfc_path}")
        return str(repo_hef_path)
    except Exception as e:
        logger.error(f"  ✗ Copy failed: {e}")
        return None


def test_model_on_mimaas(model_path: str) -> Dict:
    """
    Test a HEF model on MIMaaS via flash_and_evaluate.
    Returns result dict.
    """
    from mimaas.core.pipeline import flash_and_evaluate
    from mimaas.models.board import Board
    
    model_name = Path(model_path).stem
    logger.info(f"Testing on MIMaaS: {model_name}")
    
    try:
        board = Board(**HAILO_BOARD)
        result = flash_and_evaluate(model_path, board)
        
        if result.get("status") == "flash_and_eva_successful":
            logger.info(f"  ✓ Test PASSED")
            # Extract metrics from result
            if "hailo_results" in result:
                metrics = result["hailo_results"]
                logger.info(
                    f"    FPS: {metrics.get('fps', 'N/A')}, "
                    f"RAM: {metrics.get('ram_usage_mb', 'N/A')}MB, "
                    f"ROM: {metrics.get('rom_usage_mb', 'N/A')}MB"
                )
        else:
            logger.error(f"  ✗ Test FAILED: {result.get('error', 'Unknown error')}")
        
        return {
            "model": model_name,
            "path": model_path,
            "status": result.get("status", "unknown"),
            "result": result
        }
    except Exception as e:
        logger.error(f"  ✗ Exception: {e}")
        return {
            "model": model_name,
            "path": model_path,
            "status": "error",
            "error": str(e)
        }


def discover_models() -> Dict[str, List[str]]:
    """Discover all TFLite and HEF models in repo."""
    tflites = []
    hefs = []
    
    # Scan files/ for TFLites
    if FILES_DIR.exists():
        for f in FILES_DIR.glob("*.tflite"):
            tflites.append(str(f))
    
    # Scan hailo_models/ for HEFs
    for hm_dir in [HAILO_MODELS_DIR, COMPILED_DIR, CUSTOM_DIR]:
        if hm_dir.exists():
            for f in hm_dir.glob("*.hef"):
                hefs.append(str(f))
    
    # Remove duplicates
    hefs = list(set(hefs))
    
    logger.info(f"Discovered {len(tflites)} TFLite(s) and {len(hefs)} HEF(s)")
    return {"tflites": tflites, "hefs": hefs}


def main():
    """Main pipeline: compile TFLites and test all models."""
    
    logger.info("=" * 70)
    logger.info("Hailo Compile & Test Pipeline")
    logger.info("=" * 70)
    
    # Create output directories
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Discover models
    models = discover_models()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "compiled": [],
        "tested": [],
        "summary": {}
    }
    
    # Phase 1: Compile TFLites
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Compile TFLites")
    logger.info("=" * 70)
    
    compiled_hefs = []
    for tflite_path in models["tflites"]:
        hef_path = compile_tflite_to_hef(tflite_path)
        if hef_path:
            compiled_hefs.append(hef_path)
            results["compiled"].append({
                "tflite": tflite_path,
                "hef": hef_path,
                "status": "success"
            })
        else:
            results["compiled"].append({
                "tflite": tflite_path,
                "status": "failed"
            })
    
    logger.info(f"\nCompiled: {len(compiled_hefs)}/{len(models['tflites'])} TFLites")
    
    # Phase 2: Test all HEFs (compiled + existing)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Test All HEFs on MIMaaS")
    logger.info("=" * 70)
    
    all_hefs = compiled_hefs + models["hefs"]
    # Remove duplicates
    all_hefs = list(set(all_hefs))
    
    passed = 0
    failed = 0
    for hef_path in sorted(all_hefs):
        test_result = test_model_on_mimaas(hef_path)
        results["tested"].append(test_result)
        
        if test_result["status"] == "flash_and_eva_successful":
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTest Results: {passed} PASSED, {failed} FAILED out of {len(all_hefs)}")
    
    # Generate report
    results["summary"] = {
        "compiled_tflites": len(compiled_hefs),
        "total_tflites": len(models["tflites"]),
        "total_hefs_tested": len(all_hefs),
        "passed": passed,
        "failed": failed
    }
    
    # Save JSON results
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {RESULTS_JSON}")
    
    # Generate markdown report
    generate_report(results, models)
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info("=" * 70)


def generate_report(results: Dict, models: Dict):
    """Generate markdown report."""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Hailo Compile & Test Report\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        s = results["summary"]
        f.write(f"- **TFLites Compiled:** {s['compiled_tflites']}/{s['total_tflites']}\n")
        f.write(f"- **HEFs Tested:** {s['total_hefs_tested']}\n")
        f.write(f"- **Passed:** {s['passed']}\n")
        f.write(f"- **Failed:** {s['failed']}\n\n")
        
        # Compilation Results
        f.write("## Compilation Results\n\n")
        f.write("| TFLite | HEF | Status |\n")
        f.write("|--------|-----|--------|\n")
        for item in results["compiled"]:
            status = "✓" if item["status"] == "success" else "✗"
            hef_name = Path(item.get("hef", "N/A")).name
            f.write(f"| {Path(item['tflite']).name} | {hef_name} | {status} |\n")
        f.write("\n")
        
        # Test Results
        f.write("## Test Results\n\n")
        f.write("| Model | Status | Notes |\n")
        f.write("|-------|--------|-------|\n")
        for item in results["tested"]:
            status = "✓" if item["status"] == "flash_and_eva_successful" else "✗"
            notes = item.get("result", {}).get("note", "")
            if notes:
                notes = notes.split("\n")[0]  # First line only
            f.write(f"| {item['model']} | {status} | {notes} |\n")
        f.write("\n")
    
    logger.info(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
