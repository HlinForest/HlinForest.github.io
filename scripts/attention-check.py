"""Check article/code parity, SVG coverage and shape-to-geometry invariants."""
import ast
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
article = (root/'src/content/blog/attention-from-softmax-to-kda.md').read_text(encoding='utf-8')
reference = (root/'public/code/attention-reference.py').read_text(encoding='utf-8')
variants = (root/'public/code/attention-variants.py').read_text(encoding='utf-8')
section = article.split('<!-- implementation:start -->')[1].split('<!-- implementation:end -->')[0]
blocks = re.findall(r'```python\n(.*?)\n```', section, re.S)
expected = '"""Generated from appendix A; run with NumPy and Matplotlib. No GPU required."""\n'+'\n\n'.join(blocks)+'\n'
assert reference == expected, 'Downloadable appendix A differs from displayed code'
section_c = article.split('## 附录 C：')[1].split('## 参考资料')[0]
assert re.findall(r'```python\n(.*?)\n```', section_c, re.S)[0].strip() == variants.strip()
ast.parse(reference)
ast.parse(variants)
ledger = json.loads((root/'docs/attention-figure-ledger.json').read_text(encoding='utf-8'))
folder = root/'public/images/notes/attention-from-softmax-to-kda'
for figure in ledger:
    assert (folder/(figure['slug']+'.svg')).is_file()
    assert '/'+figure['slug']+'.svg' in article, f'Figure not linked: {figure["slug"]}'
    shape_boxes = {}
    for box in figure['boxes']:
        assert box['width'] == 27*box['cols'] and box['height'] == 27*box['rows']
        if box['rows'] == box['cols']:
            assert box['width'] == box['height']
        geometry = (box['width'], box['height'])
        assert shape_boxes.setdefault(box['shape'], geometry) == geometry, (figure['slug'], box['shape'])
assert len(re.findall(r'class="matrix-figure"', article)) == len(ledger)
print(f'PASS: code parity, Python syntax, {len(ledger)} linked SVGs, equal-shape and square geometry.')
