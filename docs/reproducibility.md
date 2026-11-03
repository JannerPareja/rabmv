# Reproducibility

Every published number can be regenerated from this repository. Synthetic results are versioned; real data and
models trained with them never enter the repository.

| Result | Command (from the repository root) | Time |
|---|---|---|
| Unit, regression and integration tests | `python -m pytest` | minutes |
| Simulation v1: calibration of the checks | `python -m simulation.calibrate` | minutes |
| Simulation v1: evaluation | `python -m simulation.evaluate audit`, `python -m simulation.evaluate models --processes 4`, `python -m simulation.evaluate summary` | hours |
| Simulation v2: engine calibration | `python -m simulation_v2.calibrate` | ~10 min |
| Simulation v2: evaluation | `python -m simulation_v2.run evaluation --processes 4` | ~23 h with 4 processes on a laptop |
| Simulation v2: summary, criteria, figures | `python -m simulation_v2.run summary --level evaluation` | seconds |
| Simulation v2: post-hoc diagnosis of C4 | `python -m simulation_v2.posthoc_c4` | ~15 min on a laptop |
| Example | `python examples/quickstart.py` | seconds |

**Frozen protocols.** Both simulations check SHA-256 fingerprints of their code, plan and of the library modules
they depend on (`simulation/FROZEN_PROTOCOL.json`, `simulation_v2/FROZEN_PROTOCOL.json`). Line endings are
normalized, so a Windows checkout gives the same fingerprints. The evaluations resume from their per-task
checkpoints.

**Numerical agreement across machines.** Checked by rerunning a replicate of simulation v2 (condition
`E_pi100_occupation`, seed 150) on Linux against the evaluation run on Windows. Three levels of agreement:

- **Exact.** The checks and the audits agreed exactly; the post-hoc diagnosis of C4 reproduced all 1,000
  replicates exactly.
- **About 9 significant digits.** The metrics of RabMV.
- **About 5 significant digits.** The metrics of the methods that use InterpretML's EBM, whose compiled
  library differs slightly between operating systems.

Summary statistics can therefore differ in the last digits; no criterion and no rounded number in this
documentation changes.

**Real data.** The `development/` harness runs RabMV and InterpretML's EBM on the Valledupar registry. It
needs the private data, which are not distributed, and writes only aggregates (counts 1–4 suppressed).
