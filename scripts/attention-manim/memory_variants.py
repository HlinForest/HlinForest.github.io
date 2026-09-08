"""Sharing, low-rank projection, discrete selection and recurrent memory."""
import numpy as np
from samples import A,N,M,F
from core_attention import product
GROUPS={}

# A routing table gives every head an explicit group without assigning new hues.
GROUPS["gqa-groups"]=[
    F("map","八个 query heads，共享两组 KV",
      [r"h=0,1,2,3\ \longrightarrow\ (K_0,V_0)",
       r"h=4,5,6,7\ \longrightarrow\ (K_1,V_1)"],
      [M("h",[["0","1","2","3"],["4","5","6","7"]],"Q"),
       r"\longrightarrow",M("g(h)",[["0","0","0","0"],["1","1","1","1"]],"K")],
      "每列槽位仍代表一个 query head；组号是离散 ID。组内共享历史 K/V，不共享 Q 或概率。"),
    product("head0","head 0 读取组 0 的 key",
            r"b_0=q_0^TK_0^T",
            N("q_0^T",[[1,0]],"Q"),N("K_0^T",[[1,0],[0,1]],"K"),
            N("b_0",[[1,0]],"P"),"得到自己的分数后，head 0 独立做缩放、mask 和 softmax。"),
    product("head1","head 1 再读取同一份 key",
            r"b_1=q_1^TK_0^T",
            N("q_1^T",[[0,1]],"Q"),N("K_0^T",[[1,0],[0,1]],"K"),
            N("b_1",[[0,1]],"P"),"K₀ 与上一帧是同一缓存对象。query 不同，所以概率可以不同。"),
    F("cache","缓存按 KV heads 计数",
      [r"\mathrm{cache}=M H_{\rm KV}(d_k+d_v)",
       r"\frac{H_Q}{H_{\rm KV}}=\frac82=4"],
      [M("H_Q",[[8]],"Q"),r"\longrightarrow",M(r"H_{\rm KV}",[[2]],"K")],
      "本例相对八头独立 KV 缓存减少为四分之一；QKᵀ、PV 的主要运算仍按八个 query heads 计算。")
]

c=A([[1,0],[0,1],[1,1]])
for h, probs in [(0,[[r"\frac{a}{a+1}",r"\frac1{a+1}"]]),
                 (1,[[r"\frac1{a+1}",r"\frac{a}{a+1}"]])]:
    GROUPS["gqa-groups"].insert(3+h, product(
        f"value-{h}",f"head {h} 的概率读取共享 V₀",
        rf"o_{h}^T=p_{h}V_0",
        M(rf"p_{h}",probs,"P"),N("V_0",np.eye(2),"V"),M(rf"o_{h}^T",probs,"O"),
        "这里 a=exp(1/√2)，与前面点积按 d_k=2 缩放后的 softmax 一致。两个 head 读同一个 V₀，得到不同输出。"))
uk=A([[1,0,1],[0,1,1]])
uv=A([[1,0],[0,2]])
qc=A([[1,1,0]])
p=A([[.25,.25,.5]])
GROUPS["mla-expand"]=[
    F("latent","先保留每个 token 的压缩表示",
      [r"C=\operatorname{RMSNorm}(XW_{DKV})"],
      [N("C",c,"S")],
      "下列数值从示意 latent C 开始，不声称是特定 RMSNorm 参数的输出。每行仍是独立历史 token，所有 heads 共享这些行。"),
    product("key","某个 head 展开内容 key",r"K_h=CU_h^K",
            N("C",c,"S"),N("U_h^K",uk,"K"),N("K_h",c@uk,"K"),
            "3×2 latent 经 2×3 投影展开为 3×3 内容 key；每个 head 有自己的升维权重。"),
    product("value","从同一 C 展开 value",r"V_h=CU_h^V",
            N("C",c,"S"),N("U_h^V",uv,"V"),N("V_h",c@uv,"V"),
            "这一步展示概念展开。推理时可用下一组结合律避免缓存展开后的每头 K/V。")
]
qbar=qc@uk.T
assert np.allclose(qc@(c@uk).T,qbar@c.T)
GROUPS["mla-absorb"]=[
    product("direct","先看展开后的内容点积",
            r"b_h=q_h^C(CU_h^K)^T",
            N("q_h^C",qc,"Q"),N("(CU_h^K)^T",(c@uk).T,"K"),
            N("b_h",qc@(c@uk).T,"P"),
            "这一写法需要展开全部历史内容 key。这里只比较未缩放内容分数。"),
    product("query","把升维权重吸收到当前 query",
            r"\bar q_h=q_h^C(U_h^K)^T",
            N("q_h^C",qc,"Q"),N("(U_h^K)^T",uk.T,"K"),N(r"\bar q_h",qbar,"Q"),
            "一次 query 变换得到 latent 宽度 2，避免对每条历史重复展开。"),
    product("cache","直接读取压缩缓存，分数相同",
            r"b_h=\bar q_h C^T",
            N(r"\bar q_h",qbar,"Q"),N("C^T",c.T,"S"),N("b_h",qbar@c.T,"P"),
            "输出与第一帧同为 [1,1,2]。吸收只改变括号，不跨过 softmax 或 RMSNorm。"),
    product("aggregate","先用主概率聚合 latent",
            r"z_h=p_hC",
            N("p_h",p,"P"),N("C",c,"S"),N("z_h",p@c,"S"),
            "为便于验算，此处另取一条示意主概率 [1/4,1/4,1/2]；它不是前一帧分数的 softmax。")
]
wo=A([[1,0],[1,1]])
GROUPS["mla-output"]=[
    product("combine","组合相邻的 value 与输出投影",
            r"W_h=U_h^VW_O^{(h)}",
            N("U_h^V",uv,"V"),N("W_O^{(h)}",wo,"W"),N("W_h",uv@wo,"W"),
            "W_O 的第 h 个行块接收该 head 的 value 输出。这里没有越过非线性运算。"),
    product("output","latent 汇总直接投影到模型宽度",
            r"\Delta x_h=z_hW_h",
            N("z_h",p@c,"S"),N("W_h",uv@wo,"W"),N(r"\Delta x_h",p@c@uv@wo,"O"),
            "每个 head 算出相同模型宽度的一份贡献；所有 head 的贡献最后相加。"),
    F("sum","对应模型通道累加各头贡献",
      [r"\Delta x=\sum_h\Delta x_h"],
      [N(r"\Delta x_0",[[2,1]],"O"),"+",N(r"\Delta x_1",[[1,2]],"O"),
       "=",N(r"\Delta x",[[3,3]],"O")],
      "本帧使用独立的两头求和小例子，强调这里相加的是投影后的模型通道，不是 latent 通道。")
]
r=A([[0,1],[-1,0]])
GROUPS["rope-pairs"]=[
    product("pair","先旋转一个二维坐标对",
            r"[x',y']=[x,y]R_{\frac{\pi}{2}}",
            N("[x,y]",[[1,0]],"Q"),N(r"R_{\frac{\pi}{2}}",r,"W"),N("[x',y']",[[0,1]],"Q"),
            "本组采用行向量右乘约定。90 度旋转把 [1,0] 变成 [0,1]。"),
    F("pairs","对每个坐标对独立旋转",
      [r"R_t=\operatorname{blockdiag}(R_{t\theta_1},R_{t\theta_2})"],
      [N("R_t",[[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]],"W")],
      "示例第一对旋转 90 度，第二对旋转 0 度；对外的零值留白。实际角度由位置乘该对频率确定。")
]
GROUPS["rotation"]=[
    product("query","同一个正交矩阵旋转 query",
            r"Q'=QR",N("Q",[[1,0]],"Q"),N("R",r,"W"),N("Q'",[[0,1]],"Q"),
            "这里 Q 与 K 使用同一个 R，区别于 RoPE 中按各自 token 位置旋转。"),
    product("key","key 也使用同一个旋转",
            r"K'=KR",N("K",[[1,1]],"K"),N("R",r,"W"),N("K'",[[-1,1]],"K"),
            "同时旋转保持点积：原点积为 1，旋转后 [0,1]·[-1,1] 仍为 1。"),
    F("cancel","中间的正交因子相消",
      [r"(QR)(KR)^T=QRR^TK^T=QK^T"],
      [N("R",r,"W"),r"\times",N("R^T",r.T,"W"),"=",N("I",np.eye(2),"W")],
      "要求 RRᵀ=I。Hadamard 旋转也需归一化；后续量化会引入额外误差，不能宣称量化后仍精确相等。")
]
GROUPS["rope-obstruction"]=[
    F("order","普通 RoPE 在投影中间插入位置因子",
      [r"(q^CR_t)(c_sU^KR_s)^T",
       r"=q^CR_tR_s^T(U^K)^Tc_s^T"],
      [N("R_0",np.eye(2),"W"),r"\qquad",N("R_1",r,"W")],
      "当前 query 的 R_t 固定，但每个历史 s 的 R_s 不同。不能把所有历史位置共用一次 query 变换。"),
    product("s0","历史位置 0 的 query 变换",
            r"u_0=q^CR_tR_0^T",
            N("q^CR_t",[[1,0]],"Q"),N("R_0^T",np.eye(2),"W"),N("u_0",[[1,0]],"Q"),
            "暂取 R_t=I；下一帧只换历史位置旋转。"),
    product("s1","历史位置 1 的变换已经不同",
            r"u_1=q^CR_tR_1^T",
            N("q^CR_t",[[1,0]],"Q"),N("R_1^T",r.T,"W"),N("u_1",[[0,-1]],"Q"),
            "一般矩阵不可交换。MLA 用独立内容与位置两路解决这个障碍。")
]
GROUPS["mla-rope"]=[
    product("content","内容路读取 latent",
            r"b^C=\bar q_hC^T",
            N(r"\bar q_h",[[1,1]],"Q"),N("C^T",c.T,"S"),N("b^C",[[1,1,2]],"P"),
            "此路可以吸收内容 key 的升维权重。"),
    product("position","位置路读取共享 RoPE key",
            r"b^R=q_h^R(K^R)^T",
            N("q_h^R",[[1,0]],"Q"),N("(K^R)^T",[[1,0,-1],[0,1,0]],"K"),N("b^R",[[1,0,-1]],"P"),
            "这里输入已经按各自位置旋转。位置 query 每头不同，位置 key 跨主 heads 共享。"),
    F("merge","两路同位置分数相加",
      [r"A_h=\frac{b^C+b^R}{\sqrt{d_c+d_R}}"],
      [N("b^C",[[1,1,2]],"P"),"+",N("b^R",[[1,0,-1]],"P"),"=",N("b^C+b^R",[[2,1,1]],"P")],
      "对每个历史位置对齐相加后缩放、mask、softmax。分母仍使用原内容宽度 d_c 加位置宽度 d_R，不改为 latent 宽度 r。")
]

GROUPS["dsa-indexer"]=[
    product("dots","索引器先算自己的点积",
            r"b_{t,s}^{(h)}=(q_t^{I,h})^Tk_s^I",
            N("(q_t^{I,h})^T",[[1,-1]],"Q"),
            N("(K^I)^T",[[1,0,2],[0,1,1]],"K"),N("b_t^{(h)}",[[1,-1,1]],"P"),
            "轻量索引器先扫描候选 key；它不是从主 Attention 输出反推索引。"),
    F("relu","每头先截去负点积",
      [r"r^{(h)}=\operatorname{ReLU}(b^{(h)})"],
      [N("b^{(h)}",[[1,-1,1]],"P"),r"\longrightarrow",N("r^{(h)}",[[1,0,1]],"P")],
      "ReLU 把负点积变为零；接下来仍有可为负的 head 权重。"),
    F("combine","加权汇总得到索引分数",
      [r"I_{t,s}=\sum_h w_{t,h}r_{t,s}^{(h)}"],
      [N("w_0r^{(0)}",[[2,0,2]],"P"),"+",N("w_1r^{(1)}",[[0,-1,-1]],"P"),
       "=",N("I",[[2,-1,1]],"P")],
      "本例权重为 2 和 −1。I 可为负，是排序分数而非概率；主 Attention 将重新计算自己的 logits。")
]
cache=A([[1,0],[0,1],[1,1],[2,0],[0,2],[2,1]])
sel=[0,2,4]; selected=cache[sel]
GROUPS["dsa-gather"]=[
    F("choose","屏蔽未来，再取 Top-K 位置",
      [r"t=4,\quad k=3",r"J_t=\operatorname{TopKIndices}(I_{t,:})"],
      [M("I",[[8,2,9,1,7,None]],"P"),r"\longrightarrow",M("J_t",[[0,2,4]],"W")],
      "位置从 0 开始。分数前三名是位置 2、0、4；图中再按源位置排序成 [0,2,4]。未来位置 5 不可选。"),
    F("gather","整数位置变成实际缓存行",
      [r"C_{\rm sel}[a,:]=C[J_t[a],:]"],
      [N("C",cache,"S",focus=[[i,j] for i in sel for j in range(2)],row_labels=[str(i) for i in range(6)]),
       r"\longrightarrow",N("C_{\\rm sel}",selected,"S",row_labels=[str(i) for i in sel])],
      "选中第 0、2、4 行，顺序对应三个 selected slots。Kᴿ 必须使用同一组位置 gather；所有主 heads 共享这组整数位置。"),
    product("main","主 query 对选中内容重新打分",
            r"b_h^C=\bar q_hC_{\rm sel}^T",
            N(r"\bar q_h",[[1,1]],"Q"),N("C_{\\rm sel}^T",selected.T,"S"),
            N("b_h^C",[[1,2,2]],"P"),"再加同位置的 RoPE 分数，并由主 softmax 归一化；索引器分数不直接作为主权重。"),
    product("read","主概率读取选中的 value 信息",
            r"z_h=p_{h,\rm sel}C_{\rm sel}",
            N("p_{h,\\rm sel}",[[.25,.25,.5]],"P"),N("C_{\\rm sel}",selected,"S"),
            N("z_h",[[.5,1.25]],"S"),
            "此处用示意主概率验算 gather 后的聚合。latent 汇总随后进入 value/output 投影。")
]
GROUPS["dsa-teacher"]=[
    F("teacher","跨主 heads 平均得到教师概率",
      [r"p_s=\operatorname{mean}_h P_{h,s}"],
      [N("P",[[.5,.25,.25],[.25,.5,.25]],"P"),r"\longrightarrow",
       M("p",[[r"\frac38",r"\frac38",r"\frac14"]],"P")],
      "先固定同一 query、同一监督位置集合。教师概率 stop-gradient，不通过该 KL 更新主模型。"),
    F("student","索引分数在同一集合上归一化",
      [r"\pi=\operatorname{softmax}(I)"],
      [N("I",[[0,0,0]],"P"),r"\longrightarrow",
       M(r"\pi",[[r"\frac13",r"\frac13",r"\frac13"]],"P")],
      "训练索引器时用 softmax 得到学生概率；推理选位置时直接用索引分数排名。"),
    F("kl","用教师分布监督索引器",
      [r"L_I=\sum_s p_s\log\frac{p_s}{\pi_s}"],
      [M("p",[[r"\frac38",r"\frac38",r"\frac14"]],"P"),
       r"\longrightarrow",M(r"\pi",[[r"\frac13",r"\frac13",r"\frac13"]],"P")],
      "warmup 的监督集合是全部可见位置；稀疏阶段是所选位置。主模型用语言模型损失，索引器用 KL，索引器输入也从主模型梯度分离。")
]

kk=A([[1,0],[0,1],[1,1]])
vv=A([[1,0],[0,1],[1,1]])
qq=A([[1,1],[1,0]])
ss=kk.T@vv
GROUPS["linear-association"]=[
    product("explicit","显式路径先生成所有相似度",
            r"A=\bar Q\bar K^T",N(r"\bar Q",qq,"Q"),N(r"\bar K^T",kk.T,"K"),
            N("A",qq@kk.T,"P"),"这里使用非负特征内积，不是 softmax。相似度矩阵仍随 query×key 数增长。"),
    product("state","改变括号，先汇总 KV 外积",
            r"S=\bar K^TV",N(r"\bar K^T",kk.T,"K"),N("V",vv,"V"),N("S",ss,"S"),
            "收缩历史 token 轴，只留下特征×value 状态。"),
    product("read","query 读取状态，分子相同",
            r"N=\bar QS=(\bar Q\bar K^T)V",N(r"\bar Q",qq,"Q"),N("S",ss,"S"),N("N",qq@ss,"O"),
            "两条路径的分子经 NumPy 检查相同。归一化还需 z=K̄ᵀ1；不能把 softmax 从乘法中移过去。")
]
assert np.array_equal((qq@kk.T)@vv,qq@ss)
for t in (1,3,6):
    ct=cache[:t]
    GROUPS["linear-association"].append(F(
        f"cache-{t}",f"第 {t} 步：逐 token 缓存与固定状态",
        [rf"\mathrm{{KV}}:\ {t}\times(2+2)={4*t}",
         r"S,z:\ 2\times2+2=6"],
        [N("[K\\mid V]",np.concatenate([ct,ct],axis=1),"K",blocks=[("col",2,"KV")],column_roles=["K","K","V","V"]),
         r"\longrightarrow",N("S",ct.T@ct,"S"),r",",N("z",ct.sum(axis=0)[:,None],"Z")],
        "左侧逐行保存每个 token 的 key/value；右侧将它们叠加成同形 S,z。新 query 在左侧读取所有历史行，在右侧只读当前 S,z；这种压缩不是 softmax 的无损替代。"))

GROUPS["linear-state"]=[F("timeline","六步状态链：写入与读取",
    [r"S_t=S_{t-1}+\bar k_tv_t^T"],
    [dict(kind="timeline")],
    "下方外积沿箭头写入加号；上方 query 从更新后的状态读取分子。每个状态都是同样的 2×2；z 同时递推、最后参与归一化。手机分两段展示，重复的 S₃ 表示同一个衔接状态。")]
s=np.zeros((2,2)); z=np.zeros((2,1))
for t in range(6):
    kt=cache[t,:,None]; vt=cache[t,None,:]
    prev=s.copy(); s+=kt@vt; z+=kt
    GROUPS["linear-state"].append(
        F(f"write-{t+1}",f"时间链：S{t} → S{t+1}",
          [rf"S_{t+1}=S_{t}+\bar k_{t+1}v_{t+1}^T"],
          [N(rf"S_{t}",prev,"S"),"+",N(rf"\bar k_{t+1}v_{t+1}^T",kt@vt,"S"),
           "=",N(rf"S_{t+1}",s.copy(),"S")],
          f"第 {t+1} 个 token 写入同一个 2×2 状态。其 key/value 均为 {cache[t].tolist()}；z 同时累加 key，当前为 {z[:,0].tolist()}。"))
    GROUPS["linear-state"].append(
        F(f"read-{t+1}",f"时间链：query 读取 S{t+1}",
          [rf"o_{t+1}^T=\frac{{\bar q_{t+1}^TS_{t+1}}}{{\bar q_{t+1}^Tz_{t+1}}}"],
          [N(rf"\bar q_{t+1}^T",[[1,1]],"Q"),r"\times",N(rf"S_{t+1}",s.copy(),"S"),
           "=",N(rf"n_{t+1}^T",A([[1,1]])@s,"O")],
          f"本帧展示分子读取；分母为 {float(z.sum()):g}。沿时间依次读完 1…6 步，状态始终保持 2×2；不是六个历史 token 的无损存储。"))

GROUPS["linear-chunk"]=[
    product("history","当前块先读取旧状态",
            r"N_{\rm hist}=\bar Q_cS_{\rm before}",
            N(r"\bar Q_c",np.eye(2),"Q"),N("S_{\\rm before}",np.eye(2),"S"),
            N("N_{\\rm hist}",np.eye(2),"O"),"旧状态只汇总本块以前的 token；不能包含本块未来。"),
    product("local","本块内部保留下三角",
            r"N_{\rm local}=A_cV_c",
            N("A_c",[[1,0],[1,1]],"P"),N("V_c",np.eye(2),"V"),
            N("N_{\\rm local}",[[1,0],[1,1]],"O"),"对角线对应当前 token，下三角对应本块更早 token，右上角严格为零。"),
    F("combine","先相加分子，分母也相加",
      [r"N=N_{\rm hist}+N_{\rm local}",
       r"l=\bar Q_cz_{\rm before}+A_c\mathbf1"],
      [N("N_{\\rm hist}",np.eye(2),"O"),"+",N("N_{\\rm local}",[[1,0],[1,1]],"O"),
       "=",N("N",[[2,0],[1,2]],"O")],
      "最后逐行计算 O_i=N_i/l_i。分别归一化两个部分后再相加通常是错的。")
]
state=A([[1,2],[3,1]]);key=A([[1],[0]]);value=A([[2,1]])
pred=key.T@state;err=value-pred;updated=state+key@err
GROUPS["delta-update"]=[
    product("predict","key 读取已有预测",r"\widehat v^T=k^TS",
            N("k^T",key.T,"K"),N("S",state,"S"),N(r"\widehat v^T",pred,"V"),"key 查询已有状态；这里不是用 query 读最终输出。"),
    F("residual","目标减预测，得到残差",[r"e^T=v^T-\widehat v^T"],
      [N("v^T",value,"V"),"-",N(r"\widehat v^T",pred,"V"),"=",N("e^T",err,"V")],
      "本例残差 [1,−1]：第一通道不足，第二通道过多。"),
    product("write","只写残差外积",r"\Delta S=\beta ke^T,\quad\beta=1",
            N("k",key,"K"),N("e^T",err,"V"),N(r"\Delta S",key@err,"S"),
            "有符号残差允许增加或减少记忆；不是把完整 value 一直累加。"),
    F("update","把修正写回状态",[r"S'=S+\Delta S"],
      [N("S",state,"S"),"+",N(r"\Delta S",key@err,"S"),"=",N("S'",updated,"S")],
      "本例单位 key、β=1，写入后沿该 key 的预测精确变成目标 [2,1]。"),
    product("read","query 读取更新后的状态",r"o^T=q^TS'",
            N("q^T",[[1,1]],"Q"),N("S'",updated,"S"),N("o^T",[[5,2]],"O"),
            "最终输出的 query 可以与写入 key 不同；读取的是更新后的 S′。")
]
GROUPS["kda-decay"]=[
    product("forget","遗忘门按状态行缩放",r"\bar S=DS",
            N("D",[[.5,0],[0,1]],"W"),N("S",[[2,2],[3,1]],"S"),
            N(r"\bar S",[[1,1],[3,1]],"S"),"第一行衰减一半，第二行保留。D 为对角矩阵，非对角零值留白。"),
    product("predict","在遗忘后的状态上预测",r"\widehat v^T=k^T\bar S",
            N("k^T",[[1,0]],"K"),N(r"\bar S",[[1,1],[3,1]],"S"),N(r"\widehat v^T",[[1,1]],"V"),
            "随后计算残差 eᵀ=vᵀ−预测，再执行前组相同的残差外积写入。"),
    F("order","合并公式时保持乘法次序",
      [r"S'=(I-\beta kk^T)DS+\beta kv^T"],
      [N("I-kk^T",[[0,0],[0,1]],"K"),r"\times",N("D",[[.5,0],[0,1]],"W")],
      "一般 key 下低秩修正与 D 不可交换；先 D 遗忘，再用 (I−βkkᵀ) 修正。DeltaNet 取 D=I，标量遗忘各行同门，KDA 各行不同门。")
]
