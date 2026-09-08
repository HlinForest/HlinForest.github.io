"""Auditable numeric storyboards; expected results are computed, not hand copied."""
import numpy as np
from spec import mat as M, frame as F

def A(a):
    return np.asarray(a, dtype=float)

def values(a):
    return [["0" if v == 0 else f"{v:g}" for v in row] for row in np.atleast_2d(a)]

def N(name, a, role, **kw):
    return M(name, values(a), role, **kw)

GROUPS = {}

# Same probability row follows the same values through each multiplication.
p = A([.25,.75])
v = A([[1,2],[5,6]])
weighted = p[:,None]*v
out = p@v
assert np.allclose(out,[4,5])
GROUPS["weighted-example"] = [
    F("select","选中一条概率行", [r"p=\left[\frac14,\frac34\right]"],
      [M("p",[[r"\frac14",r"\frac34"]],"P"),r"\longrightarrow",
       N("V",v,"V")],
      "p 的第 j 项对应 V 的第 j 行；两个数是概率，value 行包含两个输出通道。"),
]
for j in range(2):
    coefficient = r"\frac14" if j == 0 else r"\frac34"
    GROUPS["weighted-example"].append(
        F(f"scale-{j}","一个概率缩放整条 value",
          [rf"u_{j+1}^T={coefficient}v_{j+1}^T"],
          [M(rf"p_{j+1}",[[coefficient]],"P"),r"\times",
           N(rf"v_{j+1}^T",v[j],"V"),"=",N(rf"u_{j+1}^T",weighted[j],"V")],
          "同一个标量乘到这一行的每个通道；不改变向量的长度。"))
GROUPS["weighted-example"].append(
    F("add","对应通道相加，得到输出",[r"o^T=u_1^T+u_2^T"],
      [N("u_1^T",weighted[0],"V"),"+",N("u_2^T",weighted[1],"V"),"=",N("o^T",out,"O")],
      "第一个通道：0.25+3.75=4；第二个通道：0.5+4.5=5。每个输出通道都收到两条 value 的贡献。"))

# Two heads; WO is split along its rows, never assigned a different cell scale.
h1=A([[1,2]]); h2=A([[3,1]])
w1=A([[1,0],[0,1]]); w2=A([[1,1],[2,0]])
h=np.concatenate([h1,h2],axis=1); w=np.concatenate([w1,w2],axis=0)
assert np.array_equal(h@w,h1@w1+h2@w2)
GROUPS["heads"]=[
    F("heads","同一 token 的两份 head 输出",
      [r"o^{(1)T}\in\mathbb R^{1\times2}",r"o^{(2)T}\in\mathbb R^{1\times2}"],
      [N("o^{(1)T}",h1,"O"),r"\qquad",N("o^{(2)T}",h2,"O")],
      "两头对应同一个 token。head 用上标区分，颜色仍表示输出。"),
    F("concat","沿通道拼接，token 数不变",
      [r"h=[o^{(1)T}\mid o^{(2)T}]"],
      [N("h",h,"O",blocks=[("col",2,"head")])],
      "左两列属于 head 1，右两列属于 head 2；1×2 与 1×2 拼成 1×4。"),
    F("weight-blocks","输出权重按输入通道分行块",
      [r"W_O=\begin{bmatrix}W_O^{(1)}\\W_O^{(2)}\end{bmatrix}"],
      [N("W_O",w,"W",blocks=[("row",2,"head")])],
      "上两行接收 head 1，下两行接收 head 2。每块都输出相同的两个模型通道。"),
]
for j,(hj,wj) in enumerate([(h1,w1),(h2,w2)],1):
    GROUPS["heads"].append(
        F(f"project-{j}",f"head {j} 乘自己的权重行块",
          [rf"c_{j}=o^{{({j})T}}W_O^{{({j})}}"],
          [N(rf"o^{{({j})T}}",hj,"O"),r"\times",
           N(rf"W_O^{{({j})}}",wj,"W"),"=",N(rf"c_{j}",hj@wj,"O")],
          "每个输出格子由左侧向量与右侧对应列点积得到；两个 contracted 轴长度均为 2。"))
GROUPS["heads"].append(
    F("sum","两头贡献相加，完成输出投影",
      [r"\Delta x^T=c_1+c_2=hW_O"],
      [N("c_1",h1@w1,"O"),"+",N("c_2",h2@w2,"O"),"=",N(r"\Delta x^T",h@w,"O")],
      "拼接后的大矩阵乘法等于两个行块乘法之和。本例输出为 [6,5]。"))

k=A([[1,2],[2,1]]); v=A([[3,1,2],[1,4,2]])
q=A([[1,0],[1,1]])
s=np.zeros((2,3)); z=np.zeros((2,1))
GROUPS["linear-normalizer"]=[]
for t in range(2):
    kt=k[t,:,None]; vt=v[t,None,:]; qt=q[t,None,:]
    update=kt@vt; old=s.copy(); oldz=z.copy(); s+=update; z+=kt
    num=qt@s; den=qt@z; result=num/den
    prefix=f"t{t+1}"
    GROUPS["linear-normalizer"] += [
        F(prefix+"-outer",f"第 {t+1} 步：key 与 value 外积",
          [rf"\Delta S_{t+1}=\bar k_{t+1}v_{t+1}^T"],
          [N(rf"\bar k_{t+1}",kt,"K"),r"\times",N(rf"v_{t+1}^T",vt,"V"),"=",N(rf"\Delta S_{t+1}",update,"S")],
          "key 为 2×1 列向量，value 转为 1×3 行向量；输出 2×3，每个格子是对应 key 分量乘 value 分量。"),
        F(prefix+"-write",f"第 {t+1} 步：逐格累积状态",
          [rf"S_{t+1}=S_{t}+\Delta S_{t+1}"],
          [N(rf"S_{t}",old,"S"),"+",N(rf"\Delta S_{t+1}",update,"S"),"=",N(rf"S_{t+1}",s.copy(),"S")],
          "新旧状态形状相同。历史不再分 token 保存，而是叠加进同一组格子；S₀ 的零值留白。"),
        F(prefix+"-z",f"第 {t+1} 步：累积 key 特征总量",
          [rf"z_{t+1}=z_{t}+\bar k_{t+1}"],
          [N(rf"z_{t}",oldz,"Z"),"+",N(rf"\bar k_{t+1}",kt,"K"),"=",N(rf"z_{t+1}",z.copy(),"Z")],
          "z 保存 key 特征之和，维度为 2×1；它用于计算 query 对全部历史的总权重。"),
        F(prefix+"-num",f"第 {t+1} 步：query 读取分子",
          [rf"n_{t+1}^T=\bar q_{t+1}^T S_{t+1}"],
          [N(rf"\bar q_{t+1}^T",qt,"Q"),r"\times",N(rf"S_{t+1}",s.copy(),"S"),"=",N(rf"n_{t+1}^T",num,"O")],
          "query 收缩状态的 key 特征轴；每列留下一个输出通道。"),
        F(prefix+"-den",f"第 {t+1} 步：query 读取分母",
          [rf"l_{t+1}=\bar q_{t+1}^T z_{t+1}"],
          [N(rf"\bar q_{t+1}^T",qt,"Q"),r"\times",N(rf"z_{t+1}",z.copy(),"Z"),"=",N(rf"l_{t+1}",den,"Z")],
          "分母是一个标量。本例第 2 步为 1×3+1×3=6；它不是需要求逆的矩阵。"),
        F(prefix+"-normalize",f"第 {t+1} 步：每个通道除以同一分母",
          [rf"o_{t+1}^T=\frac{{n_{t+1}^T}}{{l_{t+1}}}"],
          [N(rf"n_{t+1}^T",num,"O"),r"\longrightarrow",N(rf"o_{t+1}^T",result,"O")],
          f"各通道都除以 {den.item():g}，得到 {result.tolist()[0]}。这些数直接使用非负特征 k̄、q̄；零分量用于简化示例，并非有限输入经 ELU+1 的精确输出。")
    ]
assert np.allclose(result,[[2,2.5,2]])
