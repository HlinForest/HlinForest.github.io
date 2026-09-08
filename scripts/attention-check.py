"""Check article/code parity and the complete ManimGL figure delivery."""
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
import sys
import hashlib
def source_matches(path, digest):
    # Git may check text out with LF or CRLF. Verify identical source content
    # under either newline convention; image hashes remain byte-exact.
    content=path.read_text(encoding='utf-8')
    return digest in {
        hashlib.sha256(content.encode('utf-8')).hexdigest(),
        hashlib.sha256(content.replace('\n','\r\n').encode('utf-8')).hexdigest()
    }
sys.path.insert(0,str(root/'scripts/attention-manim'))
from check_scenes import GROUPS, check
check()
manifest=json.loads((root/'docs/attention-manim-manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['sources'].items():
    assert source_matches(root/'scripts/attention-manim'/name,digest),name
folder=root/'public/images/notes/attention-from-softmax-to-kda/manim'
expected={(g,i+1) for g,frames in GROUPS.items() for i in range(len(frames))}
for layout in ['desktop','mobile']:
    rows=json.loads((folder/f'manifest-{layout}.json').read_text(encoding='utf-8'))
    assert {(r['group'],r['step']) for r in rows}==expected
    for row in rows:
        assert (folder/row['image']).is_file()
        assert hashlib.sha256((folder/row['image']).read_bytes()).hexdigest()==row['sha256']
        for name,digest in row.get('render_sources_lf',row['render_sources']).items():
            assert source_matches(root/'scripts/attention-manim'/name,digest),name
        assert row['image'] in article,row['image']
        spec=GROUPS[row['group']][row['step']-1]
        assert all(row[k]==json.loads(json.dumps(spec[k])) for k in ['key','items','tex','caption','title'])
assert article.count('data-matrix-group=')==38
assert article.count('class="matrix-step"')==len(expected)
assert 'matrix-scroll' not in article
print(f'PASS: code parity, complete manifests, {len(expected)*2} linked responsive PNGs, source hashes.')
