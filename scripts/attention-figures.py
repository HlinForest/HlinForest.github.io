"""Generate editable SVG matrix diagrams. Geometry uses one 27px edge per axis unit.

Each recipe records shapes, roles and exact structural support. The SVG is the
website-native vector source; browser QA exports PNG previews separately.
"""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/images/notes/attention-from-softmax-to-kda'
OUT.mkdir(parents=True, exist_ok=True)
PALETTE = {'q':'#4F8FA5','k':'#EE995B','v':'#8A74B5','s':'#C95B5B','p':'#4F8FA5','g':'#85898F'}
LEDGER = []

def M(name, rows, cols, shape, role='q', support='dense', values=None, parts=None):
    return dict(name=name,rows=rows,cols=cols,shape=shape,role=role,support=support,values=values,parts=parts)

def fig(slug, formula, stages, meaning):
    """Rows are algebraic lanes; every tensor face has height=rows*27,width=cols*27."""
    def width(x):
        return max(x['cols']*27+20, 142) if isinstance(x,dict) else max(48, len(x)*13+20)
    widths=[sum(width(x) for x in ops)+28*(len(ops)-1) for _,ops in stages]
    # Reserve actual prose width too; do not clip Chinese labels to a fixed canvas.
    def estimate(s, size):
        return sum(size if ord(c)>255 else size*.58 for c in s)
    w=max(820,max(widths)+64,estimate(formula,22)+64,
          max(estimate(line,16) for line in meaning)+142)
    y=92
    parts=[]
    boxes=[]
    def text(x,y,s,size=18,anchor='middle',color='#26313b'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{html.escape(s)}</text>')
    text(w/2,37,formula,22)
    for label,ops in stages:
        text(w/2,y,label,17,color='#64707b')
        y+=30
        height=max([x['rows']*27 for x in ops if isinstance(x,dict)]+[27])
        x=(w-sum(width(o) for o in ops)-28*(len(ops)-1))/2
        for op in ops:
            ww=width(op); cx=x+ww/2
            if not isinstance(op,dict):
                text(cx,y+height/2+7,op,22)
            else:
                rh,cw=op['rows']*27,op['cols']*27
                left,top=cx-cw/2,y+(height-rh)/2
                boxes.append(dict(name=op['name'],shape=op['shape'],rows=op['rows'],cols=op['cols'],x=left,y=top,width=cw,height=rh,support=op['support']))
                for i in range(op['rows']):
                    for j in range(op['cols']):
                        present = (op['support']=='dense' or op['support']=='lower' and j<=i or op['support']=='strict' and j<i or op['support']=='diag' and j==i)
                        val=op['values'][i][j] if op['values'] is not None else None
                        if val == 0 or val == '−∞': present=False
                        role=op['parts'][j] if op['parts'] else op['role']
                        color=PALETTE[role] if present else '#ffffff'
                        opacity=([.45,.68,.9][(i*7+j*11+i*j)%3] if val is None else .65) if present else 1
                        parts.append(f'<rect x="{left+j*27+1}" y="{top+i*27+1}" width="25" height="25" rx="1.5" fill="{color}" fill-opacity="{opacity}" stroke="#d7dadd" stroke-width=".5"/>')
                        if val is not None: text(left+j*27+13.5,top+i*27+18,str(val),13)
                parts.append(f'<path d="M {left-4} {top} h -4 v {rh} h 4 M {left+cw+4} {top} h 4 v {rh} h -4" fill="none" stroke="#626970"/>')
                text(cx,y+height+27,op['name'],18)
                text(cx,y+height+52,op['shape'],15,color='#64707b')
            x+=ww+28
        y+=height+103
    parts.append(f'<rect x="24" y="{y}" width="{w-48}" height="122" rx="3" fill="#f3f5f6"/>')
    for i,(label,body) in enumerate(zip(['维度','对象','机制'],meaning)):
        text(43,y+28+34*i,label,16,'start')
        text(101,y+28+34*i,body,16,'start')
    text(w/2,y+151,slug+'@五道口纳什',13,color='#7a8288')
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{y+176}" viewBox="0 0 {w} {y+176}" role="img"><title>{html.escape(formula)}</title><rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">'+''.join(parts)+'</g></svg>'
    (OUT/(slug+'.svg')).write_text(svg,encoding='utf-8')
    LEDGER.append(dict(slug=slug,formula=formula,boxes=boxes,meaning=meaning))

def mul(slug,formula,a,b,c,meaning,op='×'):
    fig(slug,formula,[('收缩：左矩阵的列，与右矩阵的行', [a,op,b,'=',c])],meaning)

mul('matmul-count','Cᵢⱼ = Σₐ Aᵢₐ Bₐⱼ',M('A',3,2,'m × k'),M('B',2,4,'k × n','k'),M('C',3,4,'m × n','s'),['m 个输出行，n 个输出列；每个格子沿 k 相乘再相加。','矩阵中一个有色格子代表一个标量；小格仅示意轴结构。','每格 k 次乘法、k−1 次加法；共有 mn 格。'])
fig('projections','Q = XWQ ; K = XWK ; V = XWV',[(name,[M('X',4,6,'N × D'), '×', M(w,6,c,'D × '+sh,role),'=',M(name,4,c,'N × '+sh,role)]) for name,w,c,sh,role in [('Q merged','WQ',4,'Hdk','q'),('K merged','WK',4,'Hdk','k'),('V merged','WV',6,'Hdv','v')]],['省略 batch；示意 H=2，dk=2，dv=3，D=6，N=4。','Q/K 是匹配特征，V 是待聚合的信息；三套权重独立学习。','每行 token 独立投影；D 被收缩，token 轴 N 保留。'])
fig('heads','split → per-head attention → concat → WO',[
('分头：按最后一维切成 H 份，再把 head 轴移到 token 轴之前',[M('Q merged',4,4,'N × (Hdk)',parts=['q','q','k','k']),'split',M('Q head 1',4,2,'N × dk'),M('Q head 2',4,2,'N × dk','k')]),
('合头：同一 token 的不同 head 输出横向拼接',[M('O head 1',4,3,'N × dv','q'), 'concat',M('O head 2',4,3,'N × dv','k'),'=',M('O merged',4,6,'N × (Hdv)',parts=['q']*3+['k']*3)]),
('输出投影：混合 head 通道，恢复模型维度',[M('O merged',4,6,'N × (Hdv)',parts=['q']*3+['k']*3),'×',M('WO',6,6,'(Hdv) × D','v'),'=',M('ΔX',4,6,'N × D','s')])],['本图 H=2，dk=2，dv=3，D=6；batch 独立重复此过程。','颜色在本图标记 head；拼接没有求和，也没有增加 token。','[B,H,N,dv] → transpose(0,2,1,3) → reshape(B,N,Hdv)。'])
mul('scores','A = QKᵀ / √dk',M('Q',4,2,'N × dk'),M('Kᵀ',2,4,'dk × M','k'),M('A',4,4,'N × M','s'),['N 个 query，M 个 key；此图 N=M=4，dk=2。','A 是实数分数，不是概率；K 转置后原来的行变成列。','一个 query 行与每一个 key 列做点积；结果再逐元素缩放。'])
mul('weighted-values','O = PV ; oᵢ = Σⱼ pᵢⱼvⱼ',M('P',4,4,'N × M','p','lower'),M('V',4,3,'M × dv','v'),M('O',4,3,'N × dv','s'),['横向权重的 M 与纵向 value 的 M 收缩，留下 N × dv。','P 是非负概率；上三角白格是因果不可见位置，权重为零。','同一行权重用于 V 的每个 value 通道；每行得到一个向量。'])
fig('weighted-example','[1/4, 3/4] V = [4, 5]', [('逐行缩放后相加',[M('p',1,2,'1 × 2','p',values=[['1/4','3/4']]),'×',M('V',2,2,'2 × 2','v',values=[[1,2],[5,6]]),'=',M('o',1,2,'1 × 2','s',values=[[4,5]])])],['2 个 key，2 个 value 通道；输入和输出都使用行向量。','第一行 V 贡献 [0.25,0.5]，第二行贡献 [3.75,4.5]。','每个输出坐标单独求和：0.25×1+0.75×5=4。'])
fig('mask-dropout','mask → row softmax → training dropout → PV',[
('mask 的 False 是禁止访问；第 3 行是无效 query',[M('keep: 1=True',3,3,'N × M','g',values=[[1,0,0],[1,1,0],[0,0,0]]),'→',M('P',3,3,'N × M','p',values=[[1,0,0],['.25','.75',0],[0,0,0]])]),
('dropout 示例：p_drop=0.5，保留的权重乘 2',[M('P 行',1,2,'1 × 2','p',values=[['.25','.75']]),'⊙',M('随机保留 / 0.5',1,2,'1 × 2','g',values=[[0,2]]),'=',M('P̃ 行',1,2,'1 × 2','p',values=[[0,'1.5']])])],['行是 query，列是 key；mask 广播到独立的 batch/head。','全遮蔽行返回全 0 是本文约定；dropout 后一行不必和为 1。','mask 决定可见性；dropout 对已有概率随机置零，再按保留率缩放。'])
fig('softmax-passes','safe softmax: max → sum exp → normalize',[
('Pass 1：读取整行，得到最大值 m=2',[M('x',1,3,'1 × M','s',values=[[0,1,2]]),'max →',M('m',1,1,'1 × 1','g',values=[[2]])]),
('Pass 2：再次读取 x，以同一 m 计算指数和 l',[M('exp(x−m)',1,3,'1 × M','p',values=[['.14','.37',1]]),'sum →',M('l',1,1,'1 × 1','g',values=[['1.5']])]),
('Pass 3：再次读取 x，重新计算指数并除以 l',[M('exp(x−m)',1,3,'1 × M','p'),'÷ l →',M('P',1,3,'1 × M','p',values=[['.09','.24','.67']])])],['一行 M 个分数；图中文字为四舍五入示意，正文有精确值。','m 是最大值，l 是缩放后的指数和；都只占一个标量。','这里不缓存指数数组：3 次读 x + 1 次写 P；pass 不等于 kernel。'])
fig('flash-tile','x = Qi Kjᵀ / √dk ; a′ = c ⊙ a + W Vj',[
('先算一个 score tile；从 HBM 载入，临时结果留在片上',[M('Qi',3,2,'R × dk'), '×',M('Kjᵀ',2,4,'dk × T','k'),'=',M('x',3,4,'R × T','s')]),
('W=exp(x−m′)，尚未除以全局分母',[M('W',3,4,'R × T','p'),'×',M('Vj',4,3,'T × dv','v'),'=',M('本块分子',3,3,'R × dv','s')]),
('c 按行广播，修正旧指数基准后再相加',[M('c',3,1,'R × 1','g'),'⊙',M('a',3,3,'R × dv','s'),'+',M('本块分子',3,3,'R × dv','s'),'=',M('a′',3,3,'R × dv','s')])],['R 条 query，T 个当前 key；m/l 各有 R 个标量，a 有 R×dv 个数。','W 是未归一化的正权重；P=W/l 需要完整历史的分母。','更新 l′=c·l+rowsum(W)；所有 key 块结束后 O=a/l。'])
mul('rotation','(QR)(KR)ᵀ = QRRᵀKᵀ = QKᵀ',M('QR',4,2,'N × dk'),M('(KR)ᵀ',2,4,'dk × M','k'),M('A unscaled',4,4,'N × M','s'),['R 为 dk × dk 正交矩阵，RRᵀ=I；两分支使用同一个 R。','旋转后的 Q/K 仍是实数特征，未做量化时点积保持。','等式只保证旋转不改点积；后续低精度量化仍会引入误差。'])
fig('linear-association','(Q̄ K̄ᵀ)V = Q̄(K̄ᵀV)',[
('路径一：先保存所有 token 对，得到 N × M',[M('Q̄',4,2,'N × r'), '×',M('K̄ᵀ',2,4,'r × M','k'),'=',M('Aκ',4,4,'N × M','p')]),
('路径一续：对 value 做加权和',[M('Aκ',4,4,'N × M','p'),'×',M('V',4,3,'M × dv','v'),'=',M('分子',4,3,'N × dv','s')]),
('路径二：先汇总历史，产生固定大小状态',[M('K̄ᵀ',2,4,'r × M','k'),'×',M('V',4,3,'M × dv','v'),'=',M('S',2,3,'r × dv','s')]),
('路径二续：每个 query 从状态中读取',[M('Q̄',4,2,'N × r'),'×',M('S',2,3,'r × dv','s'),'=',M('同一个分子',4,3,'N × dv','s')])],['Q̄=φ(Q)，K̄=φ(K)；r 为特征维度，与序列长度无关。','Aκ 是核相似度，不是 softmax 概率；两条路径使用相同的新核。','这张图是非因果分子；分母见下一图，因果要使用前缀状态。'])
fig('linear-normalizer','z = K̄ᵀ1 ; h = Q̄z ; O = numerator / (h+ε)',[
('每个特征通道累加所有 key',[M('K̄ᵀ',2,4,'r × M','k'),'×',M('1',4,1,'M × 1','g',values=[[1],[1],[1],[1]]),'=',M('z',2,1,'r × 1','k')]),
('每个 query 的总权重是一个数',[M('Q̄',4,2,'N × r'),'×',M('z',2,1,'r × 1','k'),'=',M('h',4,1,'N × 1','g')]),
('每行除以自己的总权重，广播到所有 value 通道',[M('分子',4,3,'N × dv','s'),'÷',M('h+ε',4,1,'N × 1','g'),'=',M('O',4,3,'N × dv','s')])],['1 是全 1 列向量；h 的第 i 项是第 i 个 query 的总权重。','分母归一化核权重；ε 是额外数值保护，会轻微改变精确比值。','[N,1] 自动沿 value 轴广播，不会把不同 query 混在一起。'])
fig('linear-state','St = St−1 + k̄t vtᵀ ; otᵀ = q̄tᵀ St / (q̄tᵀzt+ε)',[
('写入：列向量乘行向量，是外积',[M('k̄t',2,1,'r × 1','k',values=[[1],[2]]),'×',M('vtᵀ',1,3,'1 × dv','v',values=[[3,1,2]]),'=',M('本步写入',2,3,'r × dv','s',values=[[3,1,2],[6,2,4]])]),
('状态累加；这里 S0=0，连续读入两个 token',[M('S1',2,3,'r × dv','s',values=[[3,1,2],[6,2,4]]),'+',M('k̄2v2ᵀ',2,3,'r × dv','s',values=[[2,8,4],[1,4,2]]),'=',M('S2',2,3,'r × dv','s',values=[[5,9,6],[7,6,6]])]),
('读取：q̄2=[1,1]，z2=[3,3]，分母为 6',[M('q̄2ᵀ',1,2,'1 × r','q',values=[[1,1]]),'×',M('S2',2,3,'r × dv','s',values=[[5,9,6],[7,6,6]]),'=',M('分子',1,3,'1 × dv','s',values=[[12,15,12]])])],['第二个 key 特征为 [2,1]，value 为 [1,4,2]；示例忽略 ε。','S 的行是 key 特征通道，列是 value 通道；不再有 token 轴。','o2=[2,2.5,2]；写入 t 再读取 t，只用到前缀 1…t。'])
fig('linear-chunk','num = Q̄c Sbefore + Ac Vc ; Ac = tril(Q̄c K̄cᵀ)',[
('历史块贡献',[M('Q̄c',3,2,'C × r'),'×',M('Sbefore',2,3,'r × dv','s'),'=',M('历史分子',3,3,'C × dv','s')]),
('本块贡献；只保留对角线及下三角',[M('Ac',3,3,'C × C','p','lower'),'×',M('Vc',3,3,'C × dv','v'),'=',M('本块分子',3,3,'C × dv','s')]),
('块尾写入汇总，传给下一块',[M('K̄cᵀ',2,3,'r × C','k'),'×',M('Vc',3,3,'C × dv','v'),'+',M('Sbefore',2,3,'r × dv','s'),'=',M('Safter',2,3,'r × dv','s')])],['C 为 chunk token 数；Sbefore 汇总当前块之前的全部 token。','Ac 是非负核相似度；白格严格为 0，不可含未来 token。','总分母=Q̄c zbefore+rowsum(Ac)；先合分子分母，再做一次除法。'])
fig('delta-update','prediction = kᵀS ; eᵀ = vᵀ−prediction ; S′ = S+βkeᵀ',[
('先用 key 读取已有预测',[M('kᵀ',1,2,'1 × dk','k'),'×',M('S',2,3,'dk × dv','s'),'=',M('prediction',1,3,'1 × dv','v')]),
('只写入残差，外积不收缩 key/value 轴',[M('βk',2,1,'dk × 1','k'),'×',M('eᵀ',1,3,'1 × dv','v'),'+',M('S',2,3,'dk × dv','s'),'=',M('S′',2,3,'dk × dv','s')]),
('用 query 从更新后的状态读取输出',[M('qᵀ',1,2,'1 × dk'),'×',M('S′',2,3,'dk × dv','s'),'=',M('oᵀ',1,3,'1 × dv','s')])],['key-major 状态的行是 dk，列是 dv；q/k/v 数学上为列向量。','β 为标量写入率；e 是目标 value 与已有预测之差。','瞬时损失梯度为 −keᵀ；减去 β 倍梯度就是加上 βkeᵀ。'])
fig('kda-decay','S̄ = DS ; A = (I−βkkᵀ)D ; S′ = AS+βkvᵀ',[
('先遗忘：对角矩阵左乘，相当于逐行缩放',[M('D',2,2,'dk × dk','g','diag',values=[['.9',0],[0,'.2']]),'×',M('S',2,3,'dk × dv','s'),'=',M('S̄',2,3,'dk × dv','s')]),
('转移：低秩修正必须放在 D 左边',[M('I−βkkᵀ',2,2,'dk × dk','k'),'×',M('D',2,2,'dk × dk','g','diag'),'=',M('A',2,2,'dk × dk','s')]),
('A 是概念展开；实现用上一图的残差外积节约运算',[M('A',2,2,'dk × dk','s'),'×',M('S',2,3,'dk × dv','s'),'+',M('βkvᵀ',2,3,'dk × dv','v'),'=',M('S′',2,3,'dk × dv','s')])],['DeltaNet：D=I；GDN：对角线同一标量；KDA：每行不同门。','D 是衰减门，不是概率矩阵；对角外的白格都是严格零。','先用 S̄ 预测残差再写入；交换 D 与低秩修正会改变算法。'])
fig('chunk-solve','(I+L)U = R ; R = Diag(β)(V−Kg S0)',[
('历史预测',[M('Kg',3,2,'C × dk','k'),'×',M('S0',2,3,'dk × dv','s'),'=',M('历史预测',3,3,'C × dv','v')]),
('从目标 V 减去预测，再逐行乘 β 得到 R',[M('I+L',3,3,'C × C','k','lower'),'×',M('U',3,3,'C × dv','v'),'=',M('R',3,3,'C × dv','v')])],['L 严格下三角，对角为 0；I+L 的对角全为 1。','U 每行是修正后的实际写入 value，不是原始 V。','u1=R1；u2=R2−L21u1；u3=R3−L31u1−L32u2。'])
fig('chunk-output','O = Qg S0 + EU ; SC = Diag(g1:C)S0 + KendᵀU',[
('读取历史状态',[M('Qg',3,2,'C × dk'),'×',M('S0',2,3,'dk × dv','s'),'=',M('历史输出',3,3,'C × dv','s')]),
('读取本块写入；与上一行相加得到 O',[M('E',3,3,'C × C','p','lower'),'×',M('U',3,3,'C × dv','v'),'=',M('本块输出',3,3,'C × dv','s')]),
('将本块写入传到块尾；再加已衰减的旧状态',[M('Kendᵀ',2,3,'dk × C','k'),'×',M('U',3,3,'C × dv','v'),'=',M('块尾增量',2,3,'dk × dv','s')])],['g 是逐通道区间门乘积；Kend 每行已衰减至块尾。','E 是有符号的读取系数，不是 softmax 概率。','S_i 的展开也是此图截取前 i 行：每次写入先衰减到读取时刻。'])
mul('chunk-gram','F=(G⊙K)(K/G)ᵀ ; E=(G⊙Q)(K/G)ᵀ',M('G⊙K 或 G⊙Q',3,2,'C × dk','q'),M('(K/G)ᵀ',2,3,'dk × C','k'),M('F 或 E（mask前）',3,3,'C × C','s'),['G[i]=α1⊙…⊙αi；逐元素乘除发生在同形 C × dk 数组上。','F 截成严格下三角，E 截成含对角的下三角。','第 i,j 项的门比值为 Gi/Gj；只适用于 G 非零且数值安全。'])
fig('affine-scan','(A2,B2) ∘ (A1,B1) = (A2A1, A2B1+B2)',[
('复合转移矩阵',[M('A2',2,2,'dk × dk','k'),'×',M('A1',2,2,'dk × dk','k'),'=',M('A21',2,2,'dk × dk','s')]),
('复合写入',[M('A2',2,2,'dk × dk','k'),'×',M('B1',2,3,'dk × dv','v'),'+',M('B2',2,3,'dk × dv','v'),'=',M('B21',2,3,'dk × dv','s')])],['S 是 dk × dv；A 左乘 S，B 与 S 同形。','先执行 1 再执行 2，因此左边必须是 A2；一般不可交换。','结合律可并行组合，但 A2A1 仍需约 2dk³ FLOPs。'])
fig('layer-gates','rawα = (X Wad) Wau ; output = concat(RMSNorm(O)⊙gate) WO',[
('低秩门投影的第一步',[M('X',4,6,'N × D'),'×',M('Wad',6,2,'D × r_gate','g'),'=',M('hidden',4,2,'N × r_gate','g')]),
('第二步后 split heads，再把 log-decay 指数化',[M('hidden',4,2,'N × r_gate','g'),'×',M('Wau',2,4,'r_gate × Hdk','g'),'=',M('rawα merged',4,4,'N × Hdk','g')]),
('输出 gate 与 O 同形，按对应格子相乘',[M('RMSNorm(O)',4,3,'N × dv','s'),'⊙',M('gate（单头）',4,3,'N × dv','g'),'=',M('gated O',4,3,'N × dv','s')])],['r_gate 是门的低秩宽度，与 Linear Attention 的特征维 r 不同。','gate 是 sigmoid 输出；RMSNorm 按每行 value 通道计算均方根。','β 由 XWβ 后 sigmoid 得到；最终合头与 WO 使用前文同一拼接图。'])
fig('backward-softmax','dV=PᵀG ; dP=GVᵀ ; dQ=dAK/√dk ; dK=dAᵀQ/√dk',[
('输出对 V 的梯度',[M('Pᵀ',4,4,'M × N','p'),'×',M('G',4,3,'N × dv','s'),'=',M('dV',4,3,'M × dv','v')]),
('输出对 P 的梯度',[M('G',4,3,'N × dv','s'),'×',M('Vᵀ',3,4,'dv × M','v'),'=',M('dP',4,4,'N × M','p')]),
('softmax 局部梯度后，传回 Q',[M('dA',4,4,'N × M','p'),'×',M('K',4,2,'M × dk','k'),'=',M('dQ × √dk',4,2,'N × dk')]),
('共享 key 收到全部 query 的贡献',[M('dAᵀ',4,4,'M × N','p'),'×',M('Q',4,2,'N × dk'),'=',M('dK × √dk',4,2,'M × dk','k')])],['G=∂loss/∂O，与 O 同形；dX 表示梯度，不是模型维度 D。','图中 N=M=4；所有乘法都收缩中间维，batch/head 独立。','Flash 反向重算 tile 内 P 后，按完全相同的乘法累计梯度。'])
fig('backward-reduce','dA = P ⊙ (dP − rowsum(P⊙dP))',[
('逐元素相乘，沿 key 轴归约',[M('P⊙dP',4,4,'N × M','p'),'rowsum →',M('D',4,1,'N × 1','g')]),
('每行减去自己的 D，沿列广播',[M('dP',4,4,'N × M','p'),'−',M('D',4,1,'N × 1','g'),'=',M('centered',4,4,'N × M','s')]),
('最后乘回概率',[M('P',4,4,'N × M','p'),'⊙',M('centered',4,4,'N × M','s'),'=',M('dA',4,4,'N × M','p')])],['softmax 的归一化发生在 M 个 key 上，梯度也沿同一轴归约。','P 为 0 的位置，dA 也为 0；固定全遮蔽 query 没有梯度贡献。','Flash 可用 D=rowsum(G⊙O) 替代上图的第一次归约。'])
fig('backward-projection','dWQ = Σb Xbᵀ dQb ; dXQ = dQ WQᵀ',[
('单 batch 权重梯度；最后对 batch 求和',[M('Xᵀ',6,4,'D × N'),'×',M('dQ merged',4,4,'N × Hdk'),'=',M('dWQ per batch',6,4,'D × Hdk','k')]),
('输入梯度；Q/K/V 三条分支再相加',[M('dQ merged',4,4,'N × Hdk'),'×',M('WQᵀ',4,6,'Hdk × D','k'),'=',M('dXQ',4,6,'N × D','s')])],['先把 head 梯度合并回 N × Hdk，再对投影层求导。','权重为所有 token、所有 batch 共享，必须累加这些位置的贡献。','softmax 行 Jacobian 为 Diag(p)−ppᵀ；无需实际生成三维 Jacobian。'])
fig('backward-linear','dQ̄ = S dn + dh z ; dS += q̄ dnᵀ ; dK̄ = dS v + dz',[
('读出反向：状态将输出梯度传给 query',[M('S',2,3,'r × dv','s'),'×',M('dn',3,1,'dv × 1','v'),'=',M('dQ̄ 分子项',2,1,'r × 1')]),
('累积到状态的梯度',[M('q̄',2,1,'r × 1'),'×',M('dnᵀ',1,3,'1 × dv','v'),'=',M('dS 增量',2,3,'r × dv','s')]),
('写入反向：也要加上归一化状态梯度 dz',[M('dS',2,3,'r × dv','s'),'×',M('v',3,1,'dv × 1','v'),'=',M('dK̄ 部分',2,1,'r × 1','k')]),
('value 梯度',[M('dSᵀ',3,2,'dv × r','s'),'×',M('k̄',2,1,'r × 1','k'),'=',M('dv',3,1,'dv × 1','v')])],['dn=g/h，dh=−〈g,n〉/h²；dh 为标量，乘 z 时广播。','dS、dz 从未来向过去累积；必须使用该时刻的前缀状态。','dz+=dh·q̄；最后再乘特征映射 φ 的逐元素导数。'])
fig('backward-delta','H += qgᵀ ; dk_write=βHe ; de=βHᵀk ; dS̄=H−k deᵀ',[
('状态读出反向；dq=Sg 使用同形的矩阵×列向量',[M('q',2,1,'dk × 1'),'×',M('gᵀ',1,3,'1 × dv','v'),'=',M('H 增量',2,3,'dk × dv','s')]),
('写入反向：βHe',[M('H',2,3,'dk × dv','s'),'×',M('e',3,1,'dv × 1','v'),'=',M('dk_write / β',2,1,'dk × 1','k')]),
('残差梯度：βHᵀk',[M('Hᵀ',3,2,'dv × dk','s'),'×',M('k',2,1,'dk × 1','k'),'=',M('de / β',3,1,'dv × 1','v')]),
('预测路径从 H 中扣除的外积',[M('k',2,1,'dk × 1','k'),'×',M('deᵀ',1,3,'1 × dv','v'),'=',M('H−dS̄',2,3,'dk × dv','s')])],['dk 作为形状表示 key 维；作为带前缀的变量 dk_write 表示 key 梯度。','dβ=Σab H_ab k_a e_b；对所有状态格子的逐元素乘积求和。','dk 再减 S̄de；dα=rowsum(dS̄⊙Sprev)；Hprev=DdS̄。'])
fig('gqa-groups','GQA: head h reads KV group floor(h / group_size)',[
('组 0：Q0、Q1 使用同一份 K0，但概率各自计算',[M('Q0',3,2,'N × dk','q'),'×',M('K0ᵀ',2,3,'dk × M','k'),'=',M('A0',3,3,'N × M','p')]),
('同一个 K0 再供 Q1 读取；无须复制历史缓存',[M('Q1',3,2,'N × dk','q'),'×',M('K0ᵀ',2,3,'dk × M','k'),'=',M('A1',3,3,'N × M','p')]),
('每头自己的概率乘共享 V0，得到不同输出',[M('P0 或 P1',3,3,'N × M','p','lower'),'×',M('V0',3,3,'M × dv','v'),'=',M('O0 或 O1',3,3,'N × dv','s')])],['图示 Hq=4、Hkv=2；组 1 的 Q2/Q3 同理读取 K1/V1。','共享的是 K/V，不是 Q、概率 P 或最终输出；组内 head 仍独立。','缓存按 Hkv 计数，QK/PV 主乘法仍按 Hq 计数。'])
fig('mla-expand','C = RMSNorm(X WDKV) ; Kh = C UKh ; Vh = C UVh',[
('每个 token 压成一个所有头共用的 latent',[M('X',4,6,'M × D'),'×',M('WDKV',6,2,'D × r','g'),'norm →',M('C',4,2,'M × r','s')]),
('从同一 latent 展开某一头的内容 key',[M('C',4,2,'M × r','s'),'×',M('UKh',2,3,'r × dc','k'),'=',M('Kh',4,3,'M × dc','k')]),
('从同一 latent 展开该头的 value',[M('C',4,2,'M × r','s'),'×',M('UVh',2,3,'r × dv','v'),'=',M('Vh',4,3,'M × dv','v')])],['r 为 KV 压缩宽度；各头的 UKh、UVh 不同，C 相同。','第一行乘法后还做 RMSNorm；不能跨过归一化随意合并权重。','概念上可展开全部 K/V，但推理缓存只保存每 token 的 C。'])
fig('mla-absorb','qC(C UK)ᵀ = (qC UKᵀ)Cᵀ ; p(C UV) = (pC)UV',[
('把 key 升维转到当前 query 一侧',[M('qC',1,3,'1 × dc'),'×',M('UKᵀ',3,2,'dc × r','k'),'=',M('q̄',1,2,'1 × r')]),
('直接与缓存的 latent 比较',[M('q̄',1,2,'1 × r'),'×',M('Cᵀ',2,4,'r × M','s'),'=',M('内容分数',1,4,'1 × M','p')]),
('先加权压缩表示，再做一次 value 升维',[M('p',1,4,'1 × M','p'),'×',M('C',4,2,'M × r','s'),'=',M('z',1,2,'1 × r','s')]),
('每头得到 value 输出',[M('z',1,2,'1 × r','s'),'×',M('UV',2,3,'r × dv','v'),'=',M('o',1,3,'1 × dv','v')])],['这里只画一头、一个新 query；不物化 M 个展开后的 K/V。','C 仍保留 M 条历史；不同 head 的 p 和 z 各不相同。','吸收是同一 MLA 内部的精确结合律；并不把任意 MHA 无损压缩。'])
fig('mla-output','Δx = Σh zh (UVh WOh)',[
('模型权重固定时可以预组合两层线性投影',[M('UVh',2,3,'r × dv','v'),'×',M('WOh',3,6,'dv × D','g'),'=',M('Wh combined',2,6,'r × D','g')]),
('每头在 latent 汇总之后输出模型宽度',[M('zh',1,2,'1 × r','s'),'×',M('Wh combined',2,6,'r × D','g'),'=',M('Δxh',1,6,'1 × D','s')])],['WOh 是总 WO 对应第 h 头的 dv 行；所有头贡献最后相加。','合并矩阵可能更大，是否预存要权衡参数存储与执行成本。','图中没有跨过 softmax 或 RMSNorm；只能重排相邻线性运算。'])
fig('mla-rope','logits = (q̄ Cᵀ + qR KRᵀ) / √(dc+dR)',[
('内容路径：可以吸收，缓存 latent',[M('q̄',1,2,'1 × r'),'×',M('Cᵀ',2,4,'r × M','s'),'=',M('内容分数',1,4,'1 × M','p')]),
('位置路径：q/k 各按自己的位置旋转后点积',[M('qR',1,2,'1 × dR'),'×',M('KRᵀ',2,4,'dR × M','k'),'=',M('位置分数',1,4,'1 × M','p')]),
('相加，再缩放、mask、softmax',[M('内容分数',1,4,'1 × M','p'),'+',M('位置分数',1,4,'1 × M','p'),'=',M('未缩放总分',1,4,'1 × M','s')])],['本图 r=dR=2 仅为示意；缩放仍取原内容宽度 dc 加位置宽度 dR。','位置 key KR 跨主 heads 共享；位置 query 每头不同。','每 token 缓存 r+dR 个数；RoPE 没有破坏内容路径的吸收。'])
fig('dsa-indexer','I[t,s] = Σh w[t,h] ReLU(qI[t,h] · kI[s])',[
('轻量索引器先计算自己的分数',[M('QI 单索引头',3,2,'N × dI'),'×',M('KIᵀ（共享）',2,6,'dI × M','k'),'=',M('index dots',3,6,'N × M','s')]),
('每个头先 ReLU，再由每个 query 的 head 权重缩放',[M('ReLU(dots)',3,6,'N × M','p'),'⊙',M('wh',3,1,'N × 1','g'),'=',M('head贡献',3,6,'N × M','p')])],['索引头数 HI 与主 MLA head 数无关；各 head贡献再相加为 I。','I 是分数，不是概率；w 可以为负，所以 I 也可能为负。','QI 来自当前 query latent，KI 来自历史输入；不用先算主 Attention。'])
fig('dsa-gather','scores → TopKIndices → gather C, KR → selected MLA',[
('先屏蔽未来，再选整数位置；图中 t=4，k=3',[M('I[t,:]',1,6,'1 × M','s',values=[[8,2,9,1,7,'−∞']]),'TopK →',M('Jt',1,3,'1 × k','g','index',values=[[0,2,4]])]),
('Jt 按历史位置排序；依次取出 0、2、4 行',[M('C cache',6,2,'M × r','s'),'gather Jt →',M('Csel',3,2,'k × r','s')]),
('主 Attention 用选中内容重新打分',[M('q̄h',1,2,'1 × r'),'×',M('Cselᵀ',2,3,'r × k','s'),'=',M('content logits',1,3,'1 × k','p')]),
('加位置分数，做主 softmax 后再聚合',[M('ph,sel',1,3,'1 × k','p'),'×',M('Csel',3,2,'k × r','s'),'=',M('zh',1,2,'1 × r','s')])],['Jt 是不重复的整数源位置，范围 0…t；不是概率或主 head 轴。','同一 query 的全部主 heads 共用 Jt，主概率 ph,sel 仍各自计算。','Csel[a,:]=C[Jt[a],:]；KR 也按同一个 Jt gather，索引分数不替代主分数。'])
fig('rope-pairs','[x′, y′] = [x, y] Rθ ; RθRθᵀ = I',[
('每两个坐标旋转；不同坐标对使用不同频率',[M('坐标对',1,2,'1 × 2','q',values=[['x','y']]),'×',M('Rθ',2,2,'2 × 2','g',values=[['cos','sin'],['−sin','cos']]),'=',M('旋转后',1,2,'1 × 2','q',values=[['x′','y′']])])],['θ=position×frequency；行向量约定为右乘 Rθ。','Rθ 是旋转矩阵，不是概率；两个转置位置必须一致。','query 与 key 各用自身位置，点积由相对位置差控制。'])
fig('rope-obstruction','qC Rt Rsᵀ UKᵀ csᵀ : the middle factor depends on source position s',[
('先旋转当前 query；这里使用两维小例子',[M('qC',1,2,'1 × dc'),'×',M('Rt',2,2,'dc × dc','g'),'=',M('qC Rt',1,2,'1 × dc')]),
('历史位置 s 的旋转插在 query 与升维矩阵之间',[M('qC Rt',1,2,'1 × dc'),'×',M('Rsᵀ',2,2,'dc × dc','g'),'×',M('UKᵀ',2,3,'dc × r','k'),'=',M('依赖 s 的 query',1,3,'1 × r')])],['dc=2，r=3；Rs 随每个历史 token 改变，Rt 由当前 query 位置决定。','旋转矩阵与一般投影矩阵不可交换；改变括号不会改变这个次序。','最后再乘 csᵀ 才得到一个分数；无法对所有 s 只做一次 query 变换。'])
fig('softmax-jacobian','J = Diag(p) − ppᵀ ; Jg = p ⊙ (g − pᵀg)',[
('概率外积保留两条 key 轴',[M('p',3,1,'M × 1','p'),'×',M('pᵀ',1,3,'1 × M','p'),'=',M('ppᵀ',3,3,'M × M','p')]),
('从对角概率矩阵减去外积',[M('Diag(p)',3,3,'M × M','g','diag'),'−',M('ppᵀ',3,3,'M × M','p'),'=',M('J',3,3,'M × M','s')])],['这是单条 query 的 Jacobian；N 条 query 就要保存 N 份 M×M。','J 描述概率每一项对分数每一项的导数，不是概率矩阵。','正文的逐元素与归约形式计算 Jg，可避免物化这张矩阵。'])
fig('chunk-wy','solve (I+L)U0=Diag(β)V ; solve (I+L)W=Diag(β)Kg ; U=U0−WS0',[
('两个右端项使用同一单位下三角系数；分别前代',[M('I+L',3,3,'C × C','k','lower'),'×',M('W',3,2,'C × dk','g'),'=',M('Diag(β)Kg',3,2,'C × dk','k')]),
('把输入状态的影响从 U0 中减掉',[M('W',3,2,'C × dk','g'),'×',M('S0',2,3,'dk × dv','s'),'=',M('U0−U',3,3,'C × dv','v')])],['U0 为 C×dv，W 为 C×dk；此 W 是三角求解结果，不是投影权重。','带逆矩阵的记法代表求解；代码不需真正生成 inverse。','解的线性性允许拆分右端项；最后 U 与原三角系统解相同。'])
fig('dsa-teacher','teacher p = mean_heads(Pmain) ; loss = KL(stopgrad(p) || softmax(I))',[
('同一 query 下，跨主 heads 汇总每个历史位置的概率',[M('Pmain（逐头行）',3,4,'Hq × L','p'),'mean heads →',M('teacher p',1,4,'1 × L','p')]),
('索引分数在同一个监督集合上归一化',[M('index scores I',1,4,'1 × L','s'),'softmax →',M('student π',1,4,'1 × L','p')])],['L 是监督位置数：warmup 为全部可见 token，稀疏阶段为所选 token。','teacher 是概率，stopgrad 后固定；索引器输入同样不向主网反传。','KL=Σs p_s(log p_s−log π_s)；主模型用 LM loss，索引器用此 KL。'])
(ROOT/'docs').mkdir(exist_ok=True)
(ROOT/'docs/attention-figure-ledger.json').write_text(json.dumps(LEDGER,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Generated {len(LEDGER)} matrix diagrams and geometry ledger.')
