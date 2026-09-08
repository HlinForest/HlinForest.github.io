"""Local calculations before full matrices: projections, scores, masks and IO."""
import numpy as np
from samples import A, N, M, F, values

GROUPS={}

def product(key,title,formula,left,right,out,caption):
    return F(key,title,[formula],[left,r"\times",right,"=",out],caption)

x=A([[1,0],[0,1],[1,1]])
w=A([[1,2],[2,0]])
GROUPS["projections"]=[
    product("one","一个 token 先做一次投影",r"q_1^T=x_1^TW_Q",
            N("x_1^T",x[0],"Q"),N("W_Q",w,"W"),N("q_1^T",x[0]@w,"Q"),
            "每个输出坐标都是输入向量与投影矩阵一列的点积；W_Q 对所有 token 共享。"),
    product("all","对每一行重复，得到 Q",r"Q=XW_Q",
            N("X",x,"Q"),N("W_Q",w,"W"),N("Q",x@w,"Q"),
            "第一行与上一步完全相同。乘法只收缩输入特征轴，保留三个 token。"),
    product("keys","使用另一组权重得到 K",r"K=XW_K",
            N("X",x,"Q"),N("W_K",w.T,"W"),N("K",x@w.T,"K"),
            "Q、K、V 各自有训练得到的投影权重；它们不是从 Q 复制出来的。"),
    product("values","value 投影保留输出信息",r"V=XW_V",
            N("X",x,"Q"),N("W_V",[[1,0,1],[0,1,1]],"W"),
            N("V",x@A([[1,0,1],[0,1,1]]),"V"),
            "V 可以有三个通道，与本例两维 Q、K 不同。")
]
q=A([[1,0],[0,1],[1,1]])
k=A([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1]])
GROUPS["scores"]=[
    F("transpose","把 key 行转成参与点积的列",[r"K\in\mathbb R^{6\times2}",r"K^T\in\mathbb R^{2\times6}"],
      [N("K",k,"K"),r"\longrightarrow",N("K^T",k.T,"K")],
      "转置真实交换高和宽；原第 j 行变成第 j 列，token 编号保持不变。"),
    product("one","选 query 3 与 key 2",r"q_3^Tk_2=1\cdot0+1\cdot1=1",
            N("q_3^T",q[2],"Q"),N("k_2",k[1,:,None],"K"),N("b_{32}",[[1]],"P"),
            "两个对应分量相乘后相加，只产生一个分数。"),
    product("row","同一个 query 依次比较全部 key",r"b_3^T=q_3^TK^T",
            N("q_3^T",q[2],"Q"),N("K^T",k.T,"K"),N("b_3^T",q[2]@k.T,"P"),
            "输出的第 j 格对应历史位置 j；下一步才屏蔽未来位置。"),
    product("matrix","把 query 行排在一起",r"B=QK^T",
            N("Q",q,"Q"),N("K^T",k.T,"K"),N("B",q@k.T,"P"),
            "每条 query 独立生成一行。标准 logits 为 A=B/√d_k，这里 d_k=2。"),
    F("scale","每个点积分数使用相同缩放",
      [r"A_{ij}=\frac{B_{ij}}{\sqrt2}"],
      [N("B",q@k.T,"P"),r"\longrightarrow",
       M("A",[["0" if v==0 else rf"\frac{{{int(v)}}}{{\sqrt2}}" for v in row] for row in q@k.T],"P")],
      "缩放分母来自 key 通道数，而不是 token 数；所有图内除法使用上下分式。")
]
support=M("M",[[1 if j<=i else None for j in range(6)] for i in range(6)],"P")
support.update(row_labels=[rf"q_{i+1}" for i in range(6)],
               col_labels=[rf"k_{i+1}" for i in range(6)])
GROUPS["scores"].insert(0,F(
    "overview","六个 token：先定位 query 与 key",
    [r"M_{ij}=\mathbf1[j\leq i]"],[support],
    "总览图的格子表示因果可见性，不是概率大小。q₃ 只读取 k₁、k₂、k₃；后面的分步图再放大点积、归一化与 value 聚合。"))

GROUPS["mask-dropout"]=[
    F("causal","第 3 个 query 只能看前三个位置",
      [r"j\leq3:\ \mathrm{visible}",r"j>3:\ A_{3j}=-\infty"],
      [M("A_{3,:}",[[r"\frac1{\sqrt2}",r"\frac1{\sqrt2}",r"\frac2{\sqrt2}",None,None,None]],"P")],
      "空格表示不参与计算的未来位置。mask 应在 softmax 前把对应 logits 设为负无穷。"),
    F("empty","全遮蔽行没有可归一化的概率",
      [r"\sum_j E_{ij}=0",r"P_{i,:}=0\quad\mathrm{(convention)}"],
      [M("A_{i,:}",[[None,None,None]],"P"),r"\longrightarrow",N("P_{i,:}",[[0,0,0]],"P")],
      "先判断是否存在可见 key；全遮蔽行直接返回零。不要先执行 −∞−(−∞)，那会产生 NaN。"),
    F("drop","dropout 在概率算完后丢弃部分连接",
      [r"\widetilde P=\frac{R\odot P}{1-p_{\rm drop}}",r"p_{\rm drop}=\frac12"],
      [M("P",[[r"\frac14",r"\frac34"]],"P"),r"\longrightarrow",
       M(r"\widetilde P",[["0",r"\frac32"]],"P")],
      "示例保留掩码 R=[0,1]。保留项除以 0.5；这一行和为 1.5，不再强制归一化，期望保持原值。"),
    F("different","全遮蔽与随机全丢弃是两种情况",
      [r"R=[0,0]\Rightarrow\widetilde P=[0,0]"],
      [M("P",[[r"\frac14",r"\frac34"]],"P"),r"\longrightarrow",N(r"\widetilde P",[[0,0]],"P")],
      "这里原概率有效，只是本次训练抽样恰好全部丢弃；推理时关闭 dropout。")
]

aa=A([[1,2,1],[0,1,2]]);bb=A([[1,0],[2,1],[1,2]])
GROUPS["matmul-count"]=[
    product("entry","先数一个输出格子的运算",r"C_{11}=1\cdot1+2\cdot2+1\cdot1=6",
            N("A_{1,:}",aa[0],"Q"),N("B_{:,1}",bb[:,0,None],"K"),N("C_{11}",[[6]],"O"),
            "内积长度 k=3：需要 3 次乘法、2 次加法，共 5 FLOPs。"),
    product("matrix","输出有 m×n 个格子",r"C=AB",
            N("A",aa,"Q"),N("B",bb,"K"),N("C",aa@bb,"O"),
            "本例 m=2、n=2，共 4 个内积：4×5=20 FLOPs。"),
    F("general","推广到任意矩阵大小",
      [r"(m\times k)(k\times n)\to(m\times n)",
       r"\mathrm{FLOPs}=mn(2k-1)\approx2mkn"],
      [N("C",aa@bb,"O")],
      "一次乘加通常按 2 FLOPs 计。QKᵀ 约 2NMd_k，PV 约 2NMd_v；batch 和 head 数再乘到外面。")
]

pp=A([[1,0,0],[.5,.5,0],[.25,.25,.5]])
vv=A([[1,0],[0,2],[2,1]])
GROUPS["weighted-values"]=[
    product("row","一条概率行聚合全部 value",r"o_3^T=P_{3,:}V",
            N("P_{3,:}",pp[2],"P"),N("V",vv,"V"),N("o_3^T",pp[2]@vv,"O"),
            "第三个输出同时接收三个历史位置的贡献。"),
    product("all","所有 query 使用同一个 V",r"O=PV",
            N("P",pp,"P"),N("V",vv,"V"),N("O",pp@vv,"O"),
            "P 的行对应 query，列对应 value 的行。P 的每一行独立得到 O 的同一行；不存在 v_i 单独决定 o_i 的关系。")
]

# Values chosen so e^(x-m) is [1/4,1/2,1], avoiding ambiguous rounded sums.
GROUPS["softmax-passes"]=[
    F("max","第 1 遍：只找最大值",
      [r"x=[0,\log2,\log4]",r"m=\max_jx_j=\log4"],
      [M("x",[["0",r"\log2",r"\log4"]],"P"),r"\longrightarrow",M("m",[[r"\log4"]],"Z")],
      "最大值需要看完所有元素才能确定；这一遍不输出概率。"),
    F("sum","第 2 遍：重新读取并累加指数",
      [r"E_j=e^{x_j-m}",r"l=\sum_jE_j=\frac74"],
      [M("E",[[r"\frac14",r"\frac12","1"]],"P"),r"\longrightarrow",M("l",[[r"\frac74"]],"Z")],
      "减最大值让指数不超过 1。若不保存 E，最终输出时还需重新读取 x 并计算指数。"),
    F("prob","第 3 遍：逐项输出概率",
      [r"P_j=\frac{e^{x_j-m}}{l}"],
      [M("E",[[r"\frac14",r"\frac12","1"]],"P"),r"\longrightarrow",
       M("P",[[r"\frac17",r"\frac27",r"\frac47"]],"P")],
      "三次读取分数行。也可存 E 把计算换成额外内存搬运；遍数必须说明中间量是否保存。"),
    F("online","在线更新最大值与分母",
      [r"m'=\max(m,x)",r"l'=e^{m-m'}l+e^{x-m'}"],
      [N("l",[[1]],"Z"),r"\longrightarrow",M("l'",[[r"\frac32"]],"Z")],
      "读完 x₁=0 后 m=0,l=1；读到 x₂=log2 时旧分母乘 1/2，再加新项 1，得到 3/2。"),
    F("two-pass","在线分母仍不等于一次输出全部概率",
      [r"P_j=\frac{e^{x_j-m_{\rm final}}}{l_{\rm final}}"],
      [M("P",[[r"\frac17",r"\frac27",r"\frac47"]],"P")],
      "在线第一遍得到最终 m,l；若要输出所有概率 P，通常仍需第二遍。Flash 的关键是直接累计加权 value，避免物化整行 P。")
]
GROUPS["flash-tile"]=[
    F("first","先消费第一个 score/value",
      [r"m=0,\quad l=1,\quad a=[2,0]"],
      [N("v_1^T",[[2,0]],"V"),r"\longrightarrow",N("a",[[2,0]],"O")],
      "初始空状态由首个有效元素建立。a 是尚未除以 l 的加权和；全空 tile 必须跳过。"),
    F("rescale","最大值变大：同时重标定分子分母",
      [r"m'=\log2,\quad\gamma=e^{m-m'}=\frac12",
       r"\bar a=\gamma a,\quad\bar l=\gamma l"],
      [N("a",[[2,0]],"O"),r"\longrightarrow",N(r"\bar a",[[1,0]],"O")],
      "旧 a 从 [2,0] 变成 [1,0]，旧 l 从 1 变成 1/2。两者乘同一个因子，所以旧输出比值不变。"),
    F("append","在新尺度下加入第二条 value",
      [r"a'=\bar a+e^{x_2-m'}v_2^T",r"l'=\frac12+1=\frac32"],
      [N(r"\bar a",[[1,0]],"O"),"+",N("v_2^T",[[0,2]],"V"),"=",N("a'",[[1,2]],"O")],
      "新 score 等于新最大值，因此指数系数为 1；这一步仍不需要保存概率向量。"),
    F("tile","推广成一块 query 与一块 key",
      [r"X=\frac{Q_iK_j^T}{\sqrt{d_k}}",r"W=e^{X-m'}"],
      [M("W",[[r"\frac12","1"],["1",r"\frac12"]],"P"),r"\times",N("V_j",[[2,0],[0,2]],"V"),"=",N("WV_j",[[1,2],[2,1]],"O")],
      "每条 query 各自维护 m,l,a。W 仅在当前 tile 存在；W V_j 累计到 a，再处理下一块。"),
    F("finish","最后才做分式归一化",
      [r"o^T=\frac{a'}{l'}=\left[\frac23,\frac43\right]"],
      [N("a'",[[1,2]],"O"),r"\longrightarrow",M("o^T",[[r"\frac23",r"\frac43"]],"O")],
      "本例与对 [0,log2] 做 softmax 再乘 V 完全一致。Flash 减少中间矩阵的显存读写，主要点积计算仍随序列长度平方增长。")
]
GROUPS["flash-tile"].insert(2,F(
    "both","同一个缩放系数作用于 l 与 a",
    [r"[\bar l\mid\bar a]=\frac12[l\mid a]"],
    [N("[l\\mid a]",[[1,2,0]],"S",blocks=[("col",1,"l")]),
     r"\longrightarrow",N(r"[\bar l\mid\bar a]",[[.5,1,0]],"S",blocks=[("col",1,"l")])],
    "竖线左侧是标量分母，右侧是两通道分子。每格同时乘 1/2；旧比值 [2,0]/1 与 [1,0]/(1/2) 相同。"))
# Finish the scalar-row example before expanding into a two-query tile.
GROUPS["flash-tile"].insert(4,GROUPS["flash-tile"].pop())
GROUPS["flash-tile"] += [
    F("tile-accumulate","每条 query 累积自己的 l 与 a",
      [r"l'=\bar l+W\mathbf1",r"a'=\bar a+WV_j"],
      [N(r"[\bar l\mid\bar a]",[[.5,1,0],[.5,0,1]],"S",blocks=[("col",1,"l")]),
       "+",N("[W\\mathbf1\\mid WV_j]",[[1.5,1,2],[1.5,2,1]],"S",blocks=[("col",1,"l")]),
       "=",N("[l'\\mid a']",[[2,2,2],[2,2,2]],"S",blocks=[("col",1,"l")])],
      "竖线左边各行的分母独立累加，右边各行的分子独立累加。这里旧 m=0，新 m′=log2，旧状态已同时重标定一半。"),
    F("tile-normalize","所有 tile 完成后逐行归一化",
      [r"O_{i,:}=\frac{a'_{i,:}}{l'_i}"],
      [N("[l'\\mid a']",[[2,2,2],[2,2,2]],"S",blocks=[("col",1,"l")]),
       r"\longrightarrow",N("O_i",[[1,1],[1,1]],"O")],
      "两条 query 本例分母都是 2，各自的 [2,2] 除以自己的分母得到 [1,1]。一般情况下不同行的分母不同，不能混用。")
]
