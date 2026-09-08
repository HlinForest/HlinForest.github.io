"""Numerically check depicted operations without importing Manim or opening GL."""
import ast
import json
from pathlib import Path
import numpy as np
from samples import GROUPS as S
from core_attention import GROUPS as C
from memory_variants import GROUPS as M
from advanced import GROUPS as A

GROUPS={**S,**C,**M,**A}
ROOT=Path(__file__).resolve().parents[2]

def numbers(item):
    if not isinstance(item,dict) or "values" not in item:
        return None
    try:
        return np.array(item["values"],dtype=float)
    except (TypeError,ValueError):
        return None

def check():
    old=json.loads((ROOT/"docs/attention-figure-ledger.json").read_text(encoding="utf-8"))
    assert set(GROUPS)=={g["slug"] for g in old}
    count=0
    for group,frames in GROUPS.items():
        assert len({f["key"] for f in frames})==len(frames)
        for f in frames:
            assert f["caption"]
            for item in f["items"]:
                if isinstance(item,dict) and "symbol" in item:
                    assert not any(ord(c)<32 for c in item["symbol"]),(group,f["key"],repr(item["symbol"]))
            for formula in f["tex"]:
                assert "/" not in formula,(group,f["key"])
            for item in f["items"]:
                if isinstance(item,dict) and "values" in item:
                    assert len({len(row) for row in item["values"]})==1
            ops=f["items"]
            if len(ops)==5 and ops[3]=="=":
                left,right,expected=map(numbers,[ops[0],ops[2],ops[4]])
                if any(v is None for v in [left,right,expected]):
                    continue
                op=ops[1]
                if op==r"\times": actual=left@right
                elif op=="+": actual=left+right
                elif op=="-": actual=left-right
                elif op==r"\odot": actual=left*right
                else: continue
                assert actual.shape==expected.shape,(group,f["key"],actual.shape,expected.shape)
                np.testing.assert_allclose(actual,expected,err_msg=f"{group}:{f['key']}")
                count+=1
    for source in Path(__file__).parent.glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"))
    print(f"PASS: {len(GROUPS)} groups, {sum(map(len,GROUPS.values()))} frames, {count} numeric operations")

if __name__=="__main__":
    check()
