# MIMaaS — Hailo-10H Backend Contributions

**Author:** Maxime Karpik (maxime.karpik@grenoble-inp.org)
**Context:** Engineering project / internship at **AnkiLab** (Friedrich-Alexander-Universität Erlangen-Nürnberg), June–September 2026.

This repository showcases my personal contributions to **[MIMaaS](https://mimaas.anki.xyz)**
(Machine Intelligence with Microcontroller as a Service), an open-source platform that
evaluates neural networks on real hardware. My work added a complete backend for the
**Hailo-10H NPU** accelerator hosted on a Raspberry Pi 5.

The full project lives in a team repository. My fork with my commits is here:
**https://github.com/MaxKEdwinnn/mimaas-server_max** (branch `feature/hailo-rpi5`)

---

## What I built

| Contribution | Description |
|---|---|
| **Hailo remote inference pipeline** | End-to-end path: model → SSH/SCP transfer to Raspberry Pi 5 → `hailortcli benchmark` on the Hailo-10H → JSON metrics back to the server. |
| **DFC compilation workflow** | Automated `hailo parser → optimize → compiler` chain (TFLite → HAR → HEF) running in WSL2 from a Windows server, including Windows↔WSL path translation and non-interactive execution. |
| **Board abstraction & routing** | `hailo_rpi5` board type, YAML board configuration, and dispatch logic that routes requests to the Hailo pipeline or the original MCU/Zephyr pipeline. |
| **Benchmark & test tooling** | Batch scripts that compile and benchmark Hailo Model Zoo networks, plus unit tests for the Hailo dispatch. |
| **Compatibility study** | Systematic DFC compilation study of Model Zoo architectures, with a failure taxonomy (avgpool shift-delta quantization overflow, unsupported LayerNorm decomposition, etc.) and targeted mixed-precision workarounds (`.alls` model scripts). |
| **Documentation** | A full setup manual (Pi 5 + HailoRT + DFC in WSL2) and a technical report with benchmark results. |

## My commits

Authored by `maxime.karpik@grenoble-inp.org` on the fork:

- [`9b0c8d6`](https://github.com/MaxKEdwinnn/mimaas-server_max/commit/9b0c8d6) — feat(hailo): Hailo-10 support for Raspberry Pi 5 with remote SSH/SCP execution
- [`0ed3a7c`](https://github.com/MaxKEdwinnn/mimaas-server_max/commit/0ed3a7c) — Hailo DFC / Model Zoo workflow and runtime metrics
- [`ffdb7f8`](https://github.com/MaxKEdwinnn/mimaas-server_max/commit/ffdb7f8) — Hailo-10H backend, benchmark tools, and documentation
- [`8c8d65a`](https://github.com/MaxKEdwinnn/mimaas-server_max/commit/8c8d65a) — Hailo-10H backend and documentation (README)
- Earlier groundwork: `72dcc7b`, `8794c75`, `2991a0e` (June 2026) — initial pipeline integration and Celery board-type routing

## Repository contents

```
code-samples/   Code I wrote (each file is commented in code-samples/README.md)
docs/           Setup manual (LaTeX), technical report (LaTeX), benchmark results (CSV)
```

## Key code

- [`code-samples/run_hailo.py`](code-samples/run_hailo.py) — runner executed on the Pi 5: wraps
  `hailortcli benchmark`, parses FPS/latency/chip temperature, emits structured JSON.
- [`code-samples/demo_hailo_benchmark.py`](code-samples/demo_hailo_benchmark.py) — minimal
  end-to-end demo: board config → pipeline → remote inference → consolidated JSON report.
- [`code-samples/compile_and_test_hailo_models.py`](code-samples/compile_and_test_hailo_models.py) —
  batch TFLite→HEF compilation through the DFC in WSL2, then on-device benchmarking.
- [`code-samples/test_all_hailo_models.py`](code-samples/test_all_hailo_models.py) — regression
  runner over all compiled HEFs.

The Hailo dispatch and conversion logic I wrote lives in the main repo:
[`src/mimaas/hardware/hailo_utils.py`](https://github.com/MaxKEdwinnn/mimaas-server_max/blob/feature/hailo-rpi5/src/mimaas/hardware/hailo_utils.py)
and [`src/mimaas/core/pipeline.py`](https://github.com/MaxKEdwinnn/mimaas-server_max/blob/feature/hailo-rpi5/src/mimaas/core/pipeline.py)
(`flash_and_evaluate_hailo`).

## Results snapshot

End-to-end on Raspberry Pi 5 + Hailo-10H (from `docs/hailo_benchmark_for_excel.csv`):

| Model | Status | FPS |
|---|---|---|
| MobileNetV3-Small |  pass (with avgpool `a16_w16` workaround) | ~467 |
| EfficientNet-Lite family |  pass | up to ~167 |
| Classification / detection / segmentation HEFs | 14/14 tested | see report |

## Skills demonstrated

Python (Flask, Celery, SQLAlchemy), embedded ML deployment, Hailo DFC & HailoRT,
SSH/SCP automation, WSL2 cross-environment tooling, LaTeX technical writing, Git.
