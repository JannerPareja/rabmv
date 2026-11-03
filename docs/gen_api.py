"""Write docs/api.md from the NumPy-style docstrings of the public API (``python docs/gen_api.py``).

Sections such as *Parameters* or *Returns* become headed lists (name, type, description); free text stays as
text and blocks indented with ``::`` or ``.. math::`` become code or math blocks. The test
``tests/test_docs.py`` fails if docs/api.md is not up to date.
"""
import inspect
import re
from pathlib import Path

import rabmv

OUT = Path(__file__).resolve().parent / 'api.md'
METHODS = {'RabMVClassifier': ['fit', 'predict_proba', 'predict', 'decision_function', 'contributions'],
           'Schema': ['from_folder', 'mapping_for'], 'Catalog': ['from_toml'], 'Mapping': ['from_toml'],
           'Checks': ['alerts', 'audit'], 'Eligibility': ['matrix', 'audit'], 'Harmonized': ['summary']}
LIST_SECTIONS = {'Parameters', 'Returns', 'Attributes', 'Raises', 'Yields'}
_SECTION = re.compile(r'^-{3,}$')


def _rst_inline(text):
    """``code`` → `code` and :math:`x` → $x$."""
    text = re.sub(r':math:`([^`]+)`', r'$\1$', text)
    return re.sub(r'``([^`]+)``', r'`\1`', text)


def _split_sections(doc):
    lines = doc.splitlines()
    sections, name, buf = [], None, []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip() and _SECTION.match(lines[i + 1].strip()):
            sections.append((name, buf))
            name, buf = lines[i].strip(), []
            i += 2
            continue
        buf.append(lines[i])
        i += 1
    sections.append((name, buf))
    return sections


def _text(lines):
    """Free text: paragraphs, ``::`` literal blocks and ``.. math::`` blocks."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('.. math::'):
            block, i = [], i + 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith('    ')):
                block.append(lines[i].strip())
                i += 1
            out += ['$$', ' '.join(b for b in block if b).replace('\\\\', '\\'), '$$', '']
            continue
        if line.rstrip().endswith('::'):
            out.append(_rst_inline(line.rstrip()[:-1].rstrip() + (':' if line.strip() != '::' else '')))
            block, i = [], i + 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith('    ')):
                block.append(lines[i][4:] if lines[i].startswith('    ') else '')
                i += 1
            while block and not block[-1].strip():
                block.pop()
            while block and not block[0].strip():
                block.pop(0)
            out += ['', '```', *block, '```', '']
            continue
        out.append(_rst_inline(line))
        i += 1
    return out


def _items(lines):
    """``name : type`` followed by an indented description → a bullet per item."""
    out, current = [], None
    for line in lines:
        if line and not line.startswith(' '):
            if current:
                out.append(current)
            head = line.strip()
            name, _, kind = head.partition(' : ')
            current = [f'- **`{name.strip()}`**' + (f' (*{_rst_inline(kind.strip())}*)' if kind else ''), []]
        elif current is not None and line.strip():
            current[1].append(_rst_inline(line.strip()))
    if current:
        out.append(current)
    return [f'{head}: {" ".join(desc)}' if desc else head for head, desc in out]


def _doc(obj):
    doc = inspect.cleandoc(inspect.getdoc(obj) or '').strip()
    md = []
    for name, lines in _split_sections(doc):
        if name is None:
            md += _text(lines)
        elif name in LIST_SECTIONS:
            md += ['', f'**{name}**', '', *_items(lines), '']
        else:
            md += ['', f'**{name}**', '', *_text(lines), '']
    text = '\n'.join(md)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _signature(obj, name):
    try:
        return f'{name}{inspect.signature(obj)}'
    except (TypeError, ValueError):
        return name


def render():
    lines = ['# API', '',
             f'Reference of rabmv {rabmv.__version__}, generated from the docstrings by `docs/gen_api.py`.', '']
    lines += ['| Name | Summary |', '|---|---|']
    for name in rabmv.__all__:
        first = (inspect.getdoc(getattr(rabmv, name)) or '').strip().splitlines()[0]
        lines.append(f'| [`{name}`](#{name.lower()}) | {_rst_inline(first)} |')
    lines.append('')
    for name in rabmv.__all__:
        obj = getattr(rabmv, name)
        kind = 'class' if inspect.isclass(obj) else 'function'
        lines += [f'## {name}', '', f'*{kind}* `rabmv.{name}`', '', '```python', _signature(obj, name), '```', '',
                  _doc(obj), '']
        for m in METHODS.get(name, []):
            f = getattr(obj, m)
            lines += [f'### {name}.{m}', '', '```python', _signature(f, m), '```', '', _doc(f), '']
    return '\n'.join(lines).rstrip() + '\n'


if __name__ == '__main__':
    OUT.write_text(render(), encoding='utf-8')
    print(f'written {OUT}')
