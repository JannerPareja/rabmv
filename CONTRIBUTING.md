# Contributing to RabMV

Thank you for your interest. Bug reports, questions, schema examples from other registries and pull requests
are welcome.

## Reporting a problem

Open an issue with:

- the RabMV version (`python -c "import rabmv; print(rabmv.__version__)"`) and the versions of Python, NumPy,
  pandas and scikit-learn;
- a minimal example that reproduces it, **with synthetic data only**. Never attach real registry rows: RabMV's
  own reports contain counts only, and issues are public.

## Development

    git clone https://github.com/jannerpareja/rabmv
    cd rabmv
    python -m pip install -e ".[dev,development]"
    python -m pytest
    python -m ruff check src tests simulation simulation_v2 examples

Conventions:

- **Style.** Code, documentation and messages are in English. NumPy-style docstrings; lines of at most 120
  characters.
- **Tests.** Every change comes with tests. The regression test against EBMP v1 must stay bit-for-bit.
- **Frozen files.** Do not modify files listed in `simulation/FROZEN_PROTOCOL.json` or
  `simulation_v2/FROZEN_PROTOCOL.json`: the frozen simulations would refuse to run. New behaviour goes in new
  modules or options that are off by default (as `rabmv.refine`).
- **Documentation.** If the public API changes, regenerate `docs/api.md` with `python docs/gen_api.py` (a test
  checks it).
- **Data.** Never commit data or models trained with real data (see `.gitignore`).

## Pull requests

Describe what changes and why, link the issue, and make sure the continuous integration passes on Ubuntu and
Windows.
