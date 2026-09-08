"""Chunk solves and reverse-mode dependencies, with independently checked examples."""
import numpy as np
from samples import A,N,M,F
from core_attention import product
GROUPS={}

L=A([[1,0,0],[1,1,0],[0,1,1]])
R=A([[1,2],[3,3],[4,2]])
U=np.linalg.solve(L,R)
assert np.array_equal(U,A([[1,2],[2,1],[2,1]]))
GROUPS["chunk-solve"]=[
    F("system","块内依赖形成单位下三角系统",
      [r"(I+L)U=R"],
      [N("I+L",L,"K"),r"\times",N("U",U,"V"),"=",N("R",R,"V")],
      "图中系数矩阵含单位对角；正文 L 本身严格下三角。未知 U 逐行求出，不需要构造逆矩阵。"),
]
for i in range(3):
    contrib=L[i,:i]@U[:i] if i else np.zeros(2)
    GROUPS["chunk-solve"].append(
        F(f"row{i}",f"前代：只用已求出的第 1…{i} 行" if i else "前代：第一行没有更早依赖",
          [rf"U_{{{i+1},:}}=R_{{{i+1},:}}-\sum_{{j<{i+1}}}L_{{{i+1},j}}U_{{j,:}}"],
          [N(rf"R_{{{i+1},:}}",R[i],"V"),"-",N("L_{i,<i}U_{<i,:}",contrib,"V"),
           "=",N(rf"U_{{{i+1},:}}",U[i],"V")],
          "逐个 value 通道减去已知历史写入。每行求出后供下一行使用；这是前向代入，不是并行独立逐行除法。"))

S0=np.eye(2); Qg=A([[1,0],[1,1],[0,1]])
E=A([[1,0,0],[1,1,0],[0,1,1]])
Kend=A([[1,0],[0,1],[1,1]])
GROUPS["chunk-output"]=[
    product("history","输出先读取块前历史",
            r"O_{\rm hist}=Q_gS_0",N("Q_g",Qg,"Q"),N("S_0",S0,"S"),
            N("O_{\\rm hist}",Qg@S0,"O"),"Q_g 含从块首到当前读取时刻的门乘积。"),
    product("local","再读取本块实际写入",
            r"O_{\rm local}=EU",N("E",E,"P"),N("U",U,"V"),N("O_{\\rm local}",E@U,"O"),
            "U 使用上一组前代结果；E 含对角及下三角，是读取系数而非 softmax 概率。"),
    F("sum","同一 token、同一通道相加",
      [r"O=Q_gS_0+EU"],
      [N("Q_gS_0",Qg@S0,"O"),"+",N("EU",E@U,"O"),"=",N("O",Qg@S0+E@U,"O")],
      "历史贡献与本块贡献必须对齐同一输出行。"),
    product("tail","本块写入继续传到块尾",
            r"\Delta S_C=K_{\rm end}^TU",
            N("K_{\\rm end}^T",Kend.T,"K"),N("U",U,"V"),N(r"\Delta S_C",Kend.T@U,"S"),
            "块尾状态再加 Diag(g₁:C)S₀；K_end 已将每次写入衰减到块尾。")
]
K=A([[1,0],[1,1],[0,1]])
W=np.linalg.solve(L,K);U0=np.linalg.solve(L,R)
GROUPS["chunk-wy"]=[
    F("rhs","先求与输入状态无关的写入",
      [r"(I+L)U_0=\operatorname{Diag}(\beta)V"],
      [N("I+L",L,"K"),r"\times",N("U_0",U0,"V"),"=",N("R_V",R,"V")],
      "本例右端 R_V 已包含逐行 β 缩放；用同一个前代系数矩阵求解。"),
    F("weights","同一个系统，再解另一组右端",
      [r"(I+L)W=\operatorname{Diag}(\beta)K_g"],
      [N("I+L",L,"K"),r"\times",N("W",W,"W"),"=",N("R_K",K,"K")],
      "W 是三角求解结果，不是投影层的训练权重；它描述输入状态对本块写入的影响。"),
    F("combine","扣掉输入状态贡献，得到 U",
      [r"U=U_0-WS_0"],
      [N("U_0",U0,"V"),"-",N("WS_0",W@S0,"S"),"=",N("U",U0-W@S0,"V")],
      "线性方程的解对右端是线性的，因此可拆成两次求解。与直接求 solve(I+L,R_V−R_K S₀) 一致。")
]
assert np.allclose(U0-W@S0,np.linalg.solve(L,R-K@S0))

G=A([[1,1],[.5,1],[.5,.5]])
K=A([[1,0],[1,1],[0,1]])
left=G*K;right=K/G
gram=left@right.T
GROUPS["chunk-gram"]=[
    F("left","累计门逐格乘 key",
      [r"A=G\odot K"],
      [N("G",G,"W"),r"\odot",N("K",K,"K"),"=",N("A",left,"K")],
      "G 的每行是从块首到该位置的逐通道门乘积；乘法不收缩轴。"),
    F("right","另一侧逐格除以累计门",
      [r"B_{ia}=\frac{K_{ia}}{G_{ia}}"],
      [N("K",K,"K"),r"\longrightarrow",N("B",right,"K")],
      "分母与分子逐格对应，并非矩阵逆。此展开仅适用于 G 非零且除法数值安全。"),
    product("gram","乘转置后形成时间对的门比值",
            r"F=AB^T",N("A",left,"K"),N("B^T",right.T,"K"),N("F",gram,"P"),
            "第 i,j 格包含 G_i/G_j，表示从写入 j 到读取 i 的区间衰减。"),
    F("mask","按用途保留正确三角部分",
      [r"L:\ i>j,\qquad E:\ i\geq j"],
      [N(r"\operatorname{tril}(F,-1)",np.tril(gram,-1),"P"),
       r"\qquad",N(r"\operatorname{tril}(F)",np.tril(gram),"P")],
      "块内写入依赖严格排除对角，读取允许当前 token；两种 mask 不能互换。")
]

a1=A([[1,1],[0,1]]);a2=A([[1,0],[1,1]])
b1=np.eye(2);b2=A([[1,0],[0,0]])
GROUPS["affine-scan"]=[
    product("transition","先 1 后 2，转移矩阵左乘",
            r"A_{21}=A_2A_1",N("A_2",a2,"W"),N("A_1",a1,"W"),N("A_{21}",a2@a1,"W"),
            "将 S₁=A₁S₀+B₁ 代入 S₂=A₂S₁+B₂，得到 A₂A₁；不能交换顺序。"),
    product("carry","第一步写入经过第二步转移",
            r"\bar B_1=A_2B_1",N("A_2",a2,"W"),N("B_1",b1,"S"),N(r"\bar B_1",a2@b1,"S"),
            "早先写入也会被后续转移作用；只把 B₁+B₂ 相加是错误的。"),
    F("write","再加上第二步新写入",
      [r"B_{21}=A_2B_1+B_2"],
      [N("A_2B_1",a2@b1,"S"),"+",N("B_2",b2,"S"),"=",N("B_{21}",a2@b1+b2,"S")],
      "最终把两步表示成 S₂=A₂₁S₀+B₂₁。此复合可结合，但通常不可交换。")
]
GROUPS["layer-gates"]=[
    product("down","门投影先经过窄通道",
            r"H=XW_{ad}",N("X",[[1,0],[0,1]],"Q"),N("W_{ad}",[[1],[2]],"W"),N("H",[[1],[2]],"W"),
            "门的低秩宽度与 Linear Attention 的特征维不是同一个参数。"),
    product("up","再展开到每个 key 通道",
            r"A_{\rm raw}=HW_{au}",N("H",[[1],[2]],"W"),N("W_{au}",[[1,2]],"W"),
            N("A_{\\rm raw}",[[1,2],[2,4]],"W"),
            "raw 值还需正文规定的门参数化，再变成有效衰减系数；不能直接当作概率。"),
    F("output","输出 gate 与 value 通道逐格相乘",
      [r"Y=\operatorname{RMSNorm}(O)\odot g"],
      [N("O_{\\rm norm}",[[1,1],[1,-1]],"O"),r"\odot",
       N("g",[[.5,.5],[.5,.5]],"W"),"=",N("Y",[[.5,.5],[.5,-.5]],"O")],
      "每条 token 行沿 value 通道归一化后门控；本例归一化值用 ε=0 示意。再合头并乘 W_O。")
]

p=A([[.5],[.5]])
outer=p@p.T;diag=np.diag(p[:,0]);jac=diag-outer
GROUPS["softmax-jacobian"]=[
    product("outer","概率外积保留两个 key 轴",
            r"J_{\rm cross}=pp^T",N("p",p,"P"),N("p^T",p.T,"P"),N("pp^T",outer,"P"),
            "一个 query 的概率向量有两个元素；外积得到两两依赖，不是概率转移矩阵。"),
    F("subtract","对角项减去交叉项",
      [r"J=\operatorname{Diag}(p)-pp^T"],
      [N("Diag(p)",diag,"P"),"-",N("pp^T",outer,"P"),"=",N("J",jac,"W")],
      "提高一个 logit 会增加它自己的概率，也会降低其他项的概率，因此非对角导数为负。"),
    product("vjp","梯度乘 Jacobian",
            r"dA=J\,dP",N("J",jac,"W"),N("dP",[[1],[0]],"P"),N("dA",[[.25],[-.25]],"P"),
            "实际实现用下一组逐元素与归约公式，避免为每条 query 保存 M×M Jacobian。")
]
P=A([[1,0],[.5,.5]]);V=A([[1,0],[0,1]]);grad=A([[1,0],[0,1]])
GROUPS["backward-softmax"]=[
    product("value","输出梯度沿概率转置传给 V",
            r"dV=P^TG",N("P^T",P.T,"P"),N("G",grad,"O"),N("dV",P.T@grad,"V"),
            "共享 value 的梯度累加全部 query 的贡献；转置把 query 轴放到被收缩的位置。"),
    product("prob","沿 value 转置传给概率",
            r"dP=GV^T",N("G",grad,"O"),N("V^T",V.T,"V"),N("dP",grad@V.T,"P"),
            "dP 的形状与 P 相同；接着经过 softmax 的局部反向得到 dA。"),
    product("query","logits 梯度传回 query",
            r"dQ=\frac{dAK}{\sqrt{d_k}}",
            N("dA",[[0,0],[-.25,.25]],"P"),N("K",np.eye(2),"K"),
            N(r"\sqrt{d_k}\,dQ",[[0,0],[-.25,.25]],"Q"),
            "图中乘积尚未除以 √d_k；公式明确最终还要缩放。"),
    product("key","key 累加全部 query 的分数梯度",
            r"dK=\frac{dA^TQ}{\sqrt{d_k}}",
            N("dA^T",[[0,-.25],[0,.25]],"P"),N("Q",np.eye(2),"Q"),
            N(r"\sqrt{d_k}\,dK",[[0,-.25],[0,.25]],"K"),
            "固定 mask 不可见位置的 dA 为零；某条 query 全遮蔽，不代表共享 key 从其他 query 收不到梯度。")
]
dp=grad@V.T
d=(P*dp).sum(axis=1,keepdims=True)
centered=dp-d
GROUPS["backward-reduce"]=[
    F("product","概率与上游梯度逐格相乘",
      [r"T=P\odot dP"],
      [N("P",P,"P"),r"\odot",N("dP",dp,"P"),"=",N("T",P*dp,"P")],
      "这一乘法不收缩维度；每个位置保留对应概率与梯度的乘积。"),
    F("reduce","沿 key 轴求和，保留单列",
      [r"D_i=\sum_jT_{ij}"],
      [N("T",P*dp,"P"),r"\longrightarrow",N("D",d,"W")],
      "每条 query 有自己的标量 D_i；keepdims=True 保留 N×1，供下一步广播。"),
    F("broadcast","每行减去自己的标量",
      [r"C_{ij}=dP_{ij}-D_i"],
      [N("dP",dp,"P"),"-",N("D",d,"W"),"=",N("C",centered,"P")],
      "单列 D 沿列广播；第 i 行所有 key 都减同一个 D_i。"),
    F("final","逐格乘回概率",
      [r"dA=P\odot C"],
      [N("P",P,"P"),r"\odot",N("C",centered,"P"),"=",N("dA",P*centered,"P")],
      "本例两行 dA 分别为 [0,0] 与 [−1/4,1/4]，与显式 Jacobian 相乘一致。")
]
X=A([[1,0],[0,1],[1,1]]);dQ=A([[1,0],[0,1],[1,0]])
GROUPS["backward-projection"]=[
    product("weight","沿 token 轴累计权重梯度",
            r"dW_Q^{(b)}=X_b^TdQ_b",N("X_b^T",X.T,"Q"),N("dQ_b",dQ,"Q"),
            N("dW_Q^{(b)}",X.T@dQ,"W"),"先合并 head 梯度，恢复投影层输出布局；权重被所有 token 共享。"),
    F("batch","权重也被不同 batch 共享",
      [r"dW_Q=\sum_b dW_Q^{(b)}"],
      [N("dW_Q^{(0)}",X.T@dQ,"W"),"+",N("dW_Q^{(1)}",np.eye(2),"W"),
       "=",N("dW_Q",X.T@dQ+np.eye(2),"W")],
      "不能保留独立 batch 轴当作最终权重梯度；必须对 batch 求和。"),
    product("input","权重转置把梯度传回输入",
            r"dX_Q=dQW_Q^T",N("dQ",dQ,"Q"),N("W_Q^T",[[1,0],[1,1]],"W"),
            N("dX_Q",dQ@A([[1,0],[1,1]]),"Q"),
            "Q、K、V 三条分支都回到同一 X，输入梯度还需把三条分支贡献相加。")
]
GROUPS["backward-linear"]=[
    F("quotient","先对分式输出求导",
      [r"o=\frac nl,\quad dn=\frac gl",
       r"dl=-\frac{g^Tn}{l^2}"],
      [N("n",[[2],[4]],"O"),r"\longrightarrow",N("dn",[[.5],[0]],"O")],
      "本例 l=2,g=[1,0]ᵀ，故 dn=[1/2,0]ᵀ、dl=−1/2。分母路径不能遗漏。"),
    product("read","分子读取向状态累积外积梯度",
            r"dS\mathrel{+}=\bar q\,dn^T",
            N(r"\bar q",[[1],[1]],"Q"),N("dn^T",[[.5,0]],"O"),
            N(r"\Delta dS",[[.5,0],[.5,0]],"S"),
            "归一化状态同时累积 dz += dl·q̄；query 梯度为 S dn + dl·z。"),
    product("key","写入反向把状态梯度传给 key",
            r"d\bar k=dS\,v+dz",
            N("dS",np.eye(2),"S"),N("v",[[2],[1]],"V"),
            N("dS\\,v",[[2],[1]],"K"),
            "图中展示第一项，最终还要加 dz。dS、dz 从未来向过去累计，必须使用对应时刻的前缀状态。"),
    product("value","转置状态梯度传给 value",
            r"dv=dS^T\bar k",
            N("dS^T",np.eye(2),"S"),N(r"\bar k",[[1],[2]],"K"),N("dv",[[1],[2]],"V"),
            "最后对 q̄、k̄ 的梯度还需乘特征映射 φ 的逐元素导数。")
]
GROUPS["backward-delta"]=[
    product("read","输出读取向状态回传外积",
            r"H\mathrel{+}=qg^T",
            N("q",[[1],[0]],"Q"),N("g^T",[[1,1]],"O"),N(r"\Delta H",[[1,1],[0,0]],"S"),
            "先累积该时刻输出对状态的贡献；H 还包含未来时间回传的状态梯度。"),
    product("key","残差写入对 key 的梯度",
            r"dk_{\rm write}=\beta He,\quad\beta=1",
            N("H",np.eye(2),"S"),N("e",[[1],[-1]],"V"),
            N("dk_{\\rm write}",[[1],[-1]],"K"),
            "这只是写入路径；key 同时参与预测，随后还需减去 S̄ de。"),
    product("residual","残差接收转置状态梯度",
            r"de=\beta H^Tk,\quad\beta=1",
            N("H^T",np.eye(2),"S"),N("k",[[1],[0]],"K"),N("de",[[1],[0]],"V"),
            "目标 value 的梯度加 de；预测值的梯度为 −de。写入率梯度 dβ=Σ_ab H_ab k_a e_b。"),
    F("prediction","扣回预测路径的状态梯度",
      [r"d\bar S=H-k\,de^T"],
      [N("H",np.eye(2),"S"),"-",N("k\\,de^T",[[1,0],[0,0]],"S"),
       "=",N(r"d\bar S",[[0,0],[0,1]],"S")],
      "若包含 KDA 遗忘，继续 H_prev=D dS̄，并逐行求 dα=rowsum(dS̄⊙S_prev)。")
]
