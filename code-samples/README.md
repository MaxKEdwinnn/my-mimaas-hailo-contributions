# Code samples

All files in this folder were written by me as part of the MIMaaS Hailo-10H backend.

| File | What it does |
|---|---|
| `run_hailo.py` | Runs **on the Raspberry Pi 5**. Executes `hailortcli benchmark <model>.hef`, parses FPS, derives latency, reads chip temperature and optional DFC memory reports, and writes a structured JSON result. Handles missing `hailortcli`, non-HEF inputs and benchmark failures explicitly — it never reports a fake success. |
| `demo_hailo_benchmark.py` | Minimal end-to-end demo of the whole pipeline: loads the `hailo_pi5` board config, calls `flash_and_evaluate()`, and exports a consolidated JSON report (MIMaaS status + HailoRT metrics). |
| `compile_and_test_hailo_models.py` | Batch pipeline: TFLite → HEF via the Hailo DFC (`parser → optimize → compiler`) executed in WSL2 from Windows, then on-device benchmarking on the Pi 5. Produces a Markdown + JSON report. |
| `test_all_hailo_models.py` | Regression runner that re-tests every compiled HEF on the Hailo-10H and aggregates pass/fail results. |

Related code in the main repository (also mine, but shared files):

- `src/mimaas/hardware/hailo_utils.py` — DFC orchestration, WSL path translation, SSH/SCP
  transfer helpers, remote inference driver. See commit `ffdb7f8`.
- `src/mimaas/core/pipeline.py` — `flash_and_evaluate()` dispatch between the MCU (Zephyr)
  and Hailo pipelines, and `flash_and_evaluate_hailo()` result normalization.
- `config/boards/hailo_pi5.yaml` — board definition (host, SSH user, NPU type, timeouts).
- `tests/unit/test_hailo_dispatch.py` — unit tests for the Hailo dispatch and report parsing.
