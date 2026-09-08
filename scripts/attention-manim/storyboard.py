"""Render real ManimGL stills. Mobile uses complete objects, never pixel crops.

All examples use column vectors unless a symbol explicitly has a transpose.
The face of an (m,n) matrix is m cells high and n cells wide. A common cell
pitch is used for every operand in a frame, including transposed operands.
"""
from pathlib import Path
import json
import hashlib
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image, ImageChops
from manimlib import *

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "public/images/notes/attention-from-softmax-to-kda/manim"
COLORS = dict(Q="#4275AC", K="#CD8247", V="#8570AD", P="#498B78",
              S="#B3893D", O="#BA6573", Z="#B3893D", W="#7D8793")
INK = "#252D35"
CELL = .56


def mat(symbol, values, role, focus=None, blocks=None):
    """values are mathematical entries; None is an unfilled masked position."""
    return dict(symbol=symbol, values=values, role=role,
                focus=focus, blocks=blocks)


def frame(key, title, tex, items, caption):
    return dict(key=key, title=title, tex=tex, items=items, caption=caption)


def tex(text, scale=.65):
    assert "/" not in text, f"Use a stacked fraction: {text}"
    return Tex(text).set_color(INK).scale(scale)


def matrix(spec):
    a = spec["values"]
    rows, cols = len(a), len(a[0])
    assert all(len(row) == cols for row in a)
    grid = VGroup()
    cells = {}
    selected = spec.get("focus")
    for i, row in enumerate(a):
        for j, value in enumerate(row):
            active = selected is None or [i, j] in selected
            color = COLORS[spec.get("column_roles",[spec["role"]]*cols)[j]]
            opacity = .30 if active else .06
            zero = value is None or str(value) in ("0", "0.0")
            square = Square(side_length=CELL-.035, stroke_width=1,
                            stroke_color="#B7BEC4", fill_color=color,
                            fill_opacity=0 if zero else opacity)
            square.move_to([(j-(cols-1)/2)*CELL, ((rows-1)/2-i)*CELL, 0])
            grid.add(square)
            cells[i,j] = square
            if value is not None:
                number = tex(str(value), .40)
                # Entry strings are short numbers or single indexed symbols.
                assert number.get_width() < CELL-.02, (spec["symbol"], value)
                assert number.get_height() < CELL-.02, (spec["symbol"], value, "height")
                number.move_to(square)
                if not active:
                    number.set_opacity(.3)
                grid.add(number)
    for i, name in enumerate(spec.get("row_labels", [])):
        grid.add(tex(name,.40).next_to(cells[i,0],LEFT,buff=.16))
    for j, name in enumerate(spec.get("col_labels", [])):
        grid.add(tex(name,.40).next_to(cells[0,j],UP,buff=.16))
    label = tex(spec["symbol"], .62).next_to(grid, DOWN, buff=.18)
    shape = tex(rf"{rows}\times {cols}", .38).next_to(label, DOWN, buff=.11)
    obj = VGroup(grid, label, shape)
    obj.matrix_face = VGroup(*cells.values())
    # A block boundary occupies the existing inter-cell gutter.
    for axis, index, name in spec.get("blocks") or []:
        if axis == "row":
            y = (rows/2-index)*CELL
            line = Line([-cols*CELL/2,y,0],[cols*CELL/2,y,0],
                        color=INK, stroke_width=2)
        else:
            x = (index-cols/2)*CELL
            line = Line([x,-rows*CELL/2,0],[x,rows*CELL/2,0],
                        color=INK, stroke_width=2)
        obj.add(line)
    return obj


def timeline(mobile):
    """Actual write/read dependency graph. Repeated S3 marks continuation."""
    def segment(start,end):
        group=VGroup()
        for t in range(start,end+1):
            x=(t-start)*1.65
            state=VGroup()
            for i in range(2):
                for j in range(2):
                    square=Square(side_length=.22,stroke_width=1,
                                  stroke_color="#AAB3BC",
                                  fill_color=COLORS["S"],
                                  fill_opacity=0 if t==0 else .35)
                    square.move_to([x+(j-.5)*.25,(.5-i)*.25,0])
                    state.add(square)
            group.add(state,tex(rf"S_{t}",.5).move_to([x,-.55,0]))
            if t>start:
                plus_x=x-.825
                group.add(tex("+",.5).move_to([plus_x,0,0]))
                for a,b in [(x-1.37,plus_x-.18),(plus_x+.18,x-.28)]:
                    group.add(Arrow([a,0,0],[b,0,0],buff=0,stroke_width=1.5).set_color(INK))
                write=tex(rf"\bar k_{t}v_{t}^T",.44).move_to([plus_x,-1.25,0])
                group.add(write,Arrow([plus_x,-.94,0],[plus_x,-.2,0],
                                      buff=0,stroke_width=1.5).set_color(INK))
                read=tex(rf"\bar q_{t}^TS_{t}",.44).move_to([x,1.05,0])
                group.add(read,Arrow([x,.28,0],[x,.73,0],buff=0,
                                     stroke_width=1.5).set_color(INK))
        return group
    if mobile:
        return VGroup(segment(0,3),segment(3,6)).arrange(DOWN,buff=.75)
    return segment(0,6)


class Storyboard(Scene):
    def construct(self):
        from samples import GROUPS
        from core_attention import GROUPS as CORE
        from memory_variants import GROUPS as MEMORY
        from advanced import GROUPS as ADVANCED
        GROUPS = {**GROUPS, **CORE, **MEMORY, **ADVANCED}
        source_names=["storyboard.py","spec.py","samples.py","core_attention.py",
                      "memory_variants.py","advanced.py","custom_config.yml","requirements.txt"]
        render_sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                        for name in source_names}
        layout = os.environ.get("ATTENTION_LAYOUT", "desktop")
        wanted = set(filter(None, os.environ.get("ATTENTION_GROUPS", "").split(",")))
        mobile = layout == "mobile"
        self.camera.frame.set_shape(7 if mobile else 14, 14 if mobile else 8)
        OUT.mkdir(parents=True, exist_ok=True)
        records = []
        for group, frames in GROUPS.items():
            if wanted and group not in wanted:
                continue
            for index, spec in enumerate(frames):
                print(f"Rendering {group} {index+1} {layout}", flush=True)
                self.clear()
                title = Text(spec["title"], font="Microsoft YaHei",
                             font_size=27).set_color(INK)
                title.move_to([0, 6.5 if mobile else 3.55, 0])
                assert title.get_width() < (6.4 if mobile else 13), (spec["key"],"title")
                formulas = VGroup(*[tex(t, .65) for t in spec["tex"]])
                formulas.arrange(DOWN, buff=.16).next_to(title, DOWN, buff=.28)
                assert formulas.get_width() < (6.4 if mobile else 13), spec["key"]
                objects = []
                for item in spec["items"]:
                    objects.append(timeline(mobile) if isinstance(item,dict) and item.get("kind")=="timeline"
                                   else matrix(item) if isinstance(item, dict)
                                   else tex(r"\downarrow" if mobile and item==r"\longrightarrow" else item, .72))
                body = VGroup(*objects).arrange(DOWN if mobile else RIGHT, buff=.30 if mobile else .42)
                if not mobile:
                    for obj in objects:
                        face=getattr(obj,"matrix_face",obj)
                        obj.shift(-face.get_center()[1]*UP)
                assert body.get_width() < (6.4 if mobile else 13), spec["key"]
                body.next_to(formulas, DOWN, buff=.55)
                assert body.get_bottom()[1] > (-6.55 if mobile else -3.65), spec["key"]
                self.add(title, formulas, body)
                self.update_frame(force_draw=True)
                name = f"{group}-{index+1:02d}-{layout}.png"
                # Trim only the outer blank canvas, keeping every complete
                # object and a generous margin. Mobile objects were laid out
                # independently above; no content is sliced or reassembled.
                rendered = self.get_image().convert("RGB")
                bbox = ImageChops.difference(
                    rendered, Image.new("RGB", rendered.size, "white")
                ).getbbox()
                x0,y0,x1,y1 = bbox
                pad = 55
                rendered.crop((max(0,x0-pad),max(0,y0-pad),
                               min(rendered.width,x1+pad),
                               min(rendered.height,y1+pad))).save(OUT/name)
                records.append(dict(group=group, step=index+1, **spec,
                                    layout=layout, image=name, render_sources=render_sources,
                                    sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()))
        (OUT/f"manifest-{layout}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
