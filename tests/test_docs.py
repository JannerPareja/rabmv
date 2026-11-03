"""The documentation stays in sync with the code."""
import importlib.util
from pathlib import Path

import rabmv

ROOT = Path(__file__).resolve().parent.parent


def test_api_page_is_up_to_date():
    spec = importlib.util.spec_from_file_location('gen_api', ROOT / 'docs' / 'gen_api.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8') == mod.render(), 'run python docs/gen_api.py'


def test_versions_agree():
    import tomllib
    meta = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    cff = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    assert meta['version'] == rabmv.__version__
    assert f'version: {rabmv.__version__}' in cff


def test_every_docs_page_in_the_navigation_exists():
    nav = (ROOT / 'mkdocs.yml').read_text(encoding='utf-8')
    for page in [line.split(':', 1)[1].strip() for line in nav.splitlines() if line.strip().endswith('.md')]:
        assert (ROOT / 'docs' / page).exists(), page
