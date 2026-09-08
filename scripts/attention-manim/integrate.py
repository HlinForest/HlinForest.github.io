"""Publish only complete, matching desktop/mobile render manifests into Markdown."""
import hashlib
import html
import json
from pathlib import Path
import re
from PIL import Image
from check_scenes import GROUPS, ROOT, check

check()
folder=ROOT/"public/images/notes/attention-from-softmax-to-kda/manim"
prefix="/images/notes/attention-from-softmax-to-kda/manim/"
expected={(g,i+1) for g,frames in GROUPS.items() for i in range(len(frames))}
records={}
for layout in ("desktop","mobile"):
    rows=json.loads((folder/f"manifest-{layout}.json").read_text(encoding="utf-8"))
    assert {(r["group"],r["step"]) for r in rows}==expected,layout
    records[layout]={(r["group"],r["step"]):r for r in rows}
    for row in rows:
        current=GROUPS[row["group"]][row["step"]-1]
        for key in ("key","tex","items","caption","title"):
            assert row[key]==json.loads(json.dumps(current[key])),(layout,row["group"],row["step"],key)
        assert (folder/row["image"]).is_file()
        assert hashlib.sha256((folder/row["image"]).read_bytes()).hexdigest()==row["sha256"]
        for name,digest in row["render_sources"].items():
            assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest,name
        # First prove the exact rendering source matches locally, then publish
        # canonical text hashes so Git LF/CRLF conversion cannot break audits.
        row["render_sources_lf"]={
            name:hashlib.sha256((Path(__file__).parent/name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for name in row["render_sources"]
        }
    (folder/f"manifest-{layout}.json").write_text(
        json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")

article=ROOT/"src/content/blog/attention-from-softmax-to-kda.md"
text=article.read_text(encoding="utf-8")

def markup(group):
    parts=[f"<!-- manim-group:{group}:start -->",
           f'<section class="matrix-steps" data-matrix-group="{group}" aria-label="分步矩阵图解">']
    for i,step in enumerate(GROUPS[group],1):
        desk=records["desktop"][group,i]["image"]
        mobile=records["mobile"][group,i]["image"]
        width,height=Image.open(folder/desk).size
        label=html.escape(f"步骤 {i}：{step['title']}",quote=True)
        caption=html.escape(step["caption"])
        parts.extend([
            '<figure class="matrix-step">',
            f'<a href="{prefix}{desk}" target="_blank" rel="noopener" aria-label="{label}，打开高清图">',
            '<picture>',
            f'<source media="(max-width: 768px)" srcset="{prefix}{mobile}" />',
            f'<img src="{prefix}{desk}" alt="{label}" width="{width}" height="{height}" loading="lazy" decoding="async" />',
            '</picture></a>',
            f'<figcaption><strong>{label}</strong> {caption} <a href="{prefix}{desk}" target="_blank" rel="noopener">打开高清图</a></figcaption>',
            '</figure>'])
    parts += ['</section>',f"<!-- manim-group:{group}:end -->"]
    return "\n".join(parts)

if "<!-- manim-group:" in text:
    for group in GROUPS:
        pattern=rf"<!-- manim-group:{re.escape(group)}:start -->.*?<!-- manim-group:{re.escape(group)}:end -->"
        text,count=re.subn(pattern,lambda _:markup(group),text,flags=re.S)
        assert count==1,group
else:
    seen=set()
    def replace(match):
        group=re.search(r"/([^/]+)\.svg",match.group(0))[1]
        assert group in GROUPS and group not in seen,group
        seen.add(group)
        return markup(group)
    text=re.sub(r'<figure class="matrix-figure">.*?</figure>',replace,text,flags=re.S)
    assert seen==set(GROUPS),(seen,set(GROUPS)-seen)
text=re.sub(r'updatedAt: "[^"]+"','updatedAt: "2026-09-09"',text,count=1)
intro=(
    "**分步图解：** 本文 38 组矩阵图使用 3b1b/ManimGL 1.7.2 重新渲染。"
    "先看局部向量运算，再扩展到矩阵与时间链；每步说明放在对应图片下方。"
    "手机使用独立纵向布局，点击图片可打开高清图。Q 蓝、K 橙、V 紫、概率绿、"
    "状态及分母赭色、输出粉红，head 用编号区分。"
    "教学顺序参考用户提供的截图与 [Jia-Bin Huang 视频](https://youtu.be/Y-o545eYjXM)；"
    "视频主题为 GQA/MLA/DSA，Linear 和 Flash 的推导另依据文中原论文。"
    "各组使用局部教学数值，不代表同一套端到端模型参数；示意概率和 latent 的假设见图注。"
    " [场景源码与复现说明](https://github.com/HlinForest/HlinForest.github.io/tree/main/scripts/attention-manim)"
    " · [生成清单](/images/notes/attention-from-softmax-to-kda/manim/manifest-desktop.json)\n\n"
)
if "**分步图解：**" in text:
    text=re.sub(r"\*\*分步图解：\*\*.*?\n\n",lambda _:intro,text,count=1,flags=re.S)
else:
    text,count=re.subn(r"^图解依据[^\n]*",lambda _:intro.rstrip(),text,count=1,flags=re.M)
    assert count==1,"Missing diagram introduction"
article.write_text(text,encoding="utf-8")
sources={p.name:hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
         for p in Path(__file__).parent.iterdir() if p.suffix in (".py",".yml",".txt")}
manifest=dict(renderer="3b1b/ManimGL",version="1.7.2",groups=len(GROUPS),
              frames=len(expected),layouts=["desktop","mobile"],sources=sources)
(ROOT/"docs/attention-manim-manifest.json").write_text(
    json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"Integrated {len(GROUPS)} groups, {len(expected)} frames per layout")
