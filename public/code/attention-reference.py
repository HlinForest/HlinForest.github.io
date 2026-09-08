"""Generated from appendix A; run with NumPy and Matplotlib. No GPU required."""
import sys, math, time, importlib.util
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from contextlib import contextmanager
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(20260907)
TESTS = []
def check(name, actual, expected, atol=2e-10, rtol=2e-10):
    # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    TESTS.append(name)
def require(name, condition):
    assert bool(condition), name
    TESTS.append(name)
def normalize(x):
    # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
    # NumPy：linalg.norm 沿给定轴求欧氏长度 sqrt(sum(x*x))，不是对矩阵每格取绝对值。
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
def sigmoid(x):
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：logaddexp(a,b) 稳定计算 log(exp(a)+exp(b))，避免直接指数化大正数。
    return np.exp(-np.logaddexp(0., -x))
def softplus(x):
    # NumPy：logaddexp(a,b) 稳定计算 log(exp(a)+exp(b))，避免直接指数化大正数。
    return np.logaddexp(0., x)
def heatmaps(items, title='', cmap='viridis'):
    fig, axes = plt.subplots(1, len(items), figsize=(3.1*len(items), 3.0), squeeze=False)
    for ax, (label, a) in zip(axes[0], items):
        # NumPy：asarray 将输入视为 ndarray；若输入已是合适数组，可复用底层存储。
        a = np.asarray(a)
        ax.imshow(a, cmap=cmap, aspect='auto')
        ax.set_title(label + '  ' + str(a.shape), fontsize=10)
        if a.size <= 64:
            # NumPy：ndenumerate 同时返回坐标元组与该格数值，用来为热力图加文字。
            for (i,j), val in np.ndenumerate(a):
                ax.text(j, i, f'{val:.2g}', ha='center', va='center', fontsize=8,
                        color='white', bbox=dict(facecolor='black', alpha=.2, edgecolor='none'))
        ax.set_xlabel('column'); ax.set_ylabel('row')
    fig.suptitle(title); fig.tight_layout(); plt.show()
print('Python', sys.version.split()[0], 'NumPy', np.__version__)

# 输入 a 的最后一轴为 key；固定全遮蔽行返回全零。合法分数应为有限值，mask 用 -inf。
def softmax_masked(a, keep=None):
    if keep is not None:
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        a = np.where(keep, a, -np.inf)
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    m = np.max(a, axis=-1, keepdims=True)
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    safe_m = np.where(np.isfinite(m), m, 0.)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    e = np.exp(a - safe_m)
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    den = e.sum(-1, keepdims=True)
    # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    return np.divide(e, den, out=np.zeros_like(e), where=den > 0)

# 把 query/key 的局部下标加上全局起点，返回 [n,m] 布尔可见性矩阵。
def position_mask(n, m, q_start=0, k_start=0):
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    return (np.arange(m) + k_start)[None,:] <= (np.arange(n) + q_start)[:,None]

# q/k/v 前导轴是 batch/head；输出 o[...,N,dv] 与概率 p[...,N,M]。
def softmax_attention(q, k, v, causal=False, keep=None, q_start=0):
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2]
    assert q.shape[-1] == k.shape[-1] and k.shape[-2] == v.shape[-2]
    if causal:
        cm = position_mask(q.shape[-2], k.shape[-2], q_start)
        keep = cm if keep is None else (keep & cm)
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    p = softmax_masked((q @ k.swapaxes(-1,-2)) / math.sqrt(q.shape[-1]), keep)
    return p @ v, p

# 先在通道内部切出 head，再交换 N/H：BN(Hd) → BN H d → BH N d。
def split_heads(x, h):
    b,n,w = x.shape
    assert w % h == 0
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    # NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
    return x.reshape(b,n,h,w//h).transpose(0,2,1,3)

# 先把 token 轴移回 head 前，再合并 head×通道；它与 split_heads 互为逆操作。
def merge_heads(x):
    b,h,n,d = x.shape
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    # NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
    return x.transpose(0,2,1,3).reshape(b,n,h*d)

# 每组 query heads 共享一套 K/V；本函数以 repeat 复制作数学参考，不代表内存优化。
def gqa(q, k, v, **kwargs):
    # 教学版复制；生产 kernel 根据 query head 寻址共享的 KV head。
    assert q.shape[1] % k.shape[1] == 0
    group = q.shape[1] // k.shape[1]
    # NumPy：repeat 沿指定轴重复元素/切片，确实会复制；这里只用于验证共享 KV 的基准。
    return softmax_attention(q, np.repeat(k,group,axis=1),
                            # NumPy：repeat 沿指定轴重复元素/切片，确实会复制；这里只用于验证共享 KV 的基准。
                            np.repeat(v,group,axis=1), **kwargs)[0]

B,H,N,DK,DV = 2,2,9,4,3
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
q = rng.normal(size=(B,H,N,DK)); k = rng.normal(size=q.shape)
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
v = rng.normal(size=(B,H,N,DV))
o,p = softmax_attention(q,k,v,causal=True)
heatmaps([('Q',q[0,0,:4]), ('K transpose',k[0,0,:4].T),
          ('P: causal rows',p[0,0,:4,:4]), ('V',v[0,0,:4]), ('O',o[0,0,:4])],
         'Q @ K.T -> row softmax -> P @ V')
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
# NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
check('softmax rows sum 1',p.sum(-1),np.ones((B,H,N)))
# NumPy：all 检查布尔数组是否全部为真，返回真/假而非数值误差大小。
# NumPy：triu 提取上三角；k=1 只保留严格未来位置，常用于检查因果输出是否为零。
require('causal upper triangle zero', np.all(np.triu(p,1)==0))
# NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
# NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
check('head split merge', merge_heads(split_heads(np.arange(120).reshape(2,5,12),3)),
      # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
      # NumPy：reshape 只重组轴形状，元素总数必须相等；不会自动完成 token/head 的交换。
      np.arange(120).reshape(2,5,12))
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
zero,_ = softmax_attention(q,k,v,keep=np.zeros((N,N),dtype=bool))
# NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
check('all masked rows zero',zero,np.zeros_like(zero))
# Prefill 与逐 token decode 必须完全一致。
# NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
decoded = np.concatenate([softmax_attention(q[:,:,t:t+1],k[:,:,:t+1],v[:,:,:t+1])[0]
                          for t in range(N)],axis=2)
check('softmax prefill equals decode',decoded,o)
check('offset mask decode',softmax_attention(q[:,:,-1:],k,v,causal=True,q_start=N-1)[0],o[:,:,-1:])

# 先经过 O=PV，再经过逐行 softmax，最后经过 scaled QK，按链式法则返回梯度。
def softmax_backward(q,k,v,p,go):
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    dv = p.swapaxes(-1,-2) @ go
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    dp = go @ v.swapaxes(-1,-2)
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    da = p * (dp - (p*dp).sum(-1,keepdims=True))
    scale = q.shape[-1] ** -0.5
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    return (da @ k)*scale, (da.swapaxes(-1,-2) @ q)*scale, dv

# 固定其他元素，仅对一个坐标做正负 eps 扰动，用中心差分检验手写梯度。
def finite_difference(fn, x, eps=1e-6):
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    x = x.copy(); grad = np.zeros_like(x)
    # NumPy：ndindex 遍历所有坐标元组；有限差分一次只扰动一个输入元素。
    for idx in np.ndindex(x.shape):
        old=x[idx]; x[idx]=old+eps; pos=fn(x)
        x[idx]=old-eps; neg=fn(x); x[idx]=old
        grad[idx]=(pos-neg)/(2*eps)
    return grad

# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
sq,sk,sv = [rng.normal(size=s) for s in [(1,1,3,2),(1,1,3,2),(1,1,3,2)]]
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
so,sp=softmax_attention(sq,sk,sv,causal=True); go=rng.normal(size=so.shape)
grads=softmax_backward(sq,sk,sv,sp,go)
for j in range(3):
    arr=[sq,sk,sv]
    def loss(z):
        args=arr.copy(); args[j]=z
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        return np.sum(softmax_attention(*args,causal=True)[0]*go)
    check('softmax finite difference '+str(j),grads[j],finite_difference(loss,arr[j]),atol=2e-8,rtol=2e-6)

# ELU+1 非负特征映射；改变核定义是算法选择，不等于保持原 softmax。
def phi(x):
    # 避免 np.where 同时计算 exp(大正数)。
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：minimum 比较对应位置取较小值；例如先将指数输入限制为不大于 0，避免溢出。
    return np.where(x>=0, x+1, np.exp(np.minimum(x,0)))

def linear_dense(q,k,v,causal=True,eps=1e-12):
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    a=phi(q) @ phi(k).swapaxes(-1,-2)
    if causal: a=a*position_mask(q.shape[-2],k.shape[-2])
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    return (a@v)/(a.sum(-1,keepdims=True)+eps)

# 每步写 k 的外积，再读 q；s[...,r,dv]、z[...,r] 可跨调用传递，返回全部 token 输出。
def linear_recurrent(q,k,v,state=None,eps=1e-12):
    q,k=phi(q),phi(k)
    shape=q.shape[:-2]; r=q.shape[-1]; dv=v.shape[-1]
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s,z=(np.zeros(shape+(r,dv)),np.zeros(shape+(r,))) if state is None else (state[0].copy(),state[1].copy())
    out=[]
    for t in range(q.shape[-2]):
        kt,vt,qt=k[...,t,:],v[...,t,:],q[...,t,:]
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        s=s+kt[..., :,None]*vt[...,None,:]; z=z+kt
        # NumPy：einsum("...r,...rv->...v")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        out.append(np.einsum('...r,...rv->...v',qt,s)/(np.sum(qt*z,-1,keepdims=True)+eps))
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    return np.stack(out,-2),(s,z)

# 历史块通过 s,z 贡献，本块通过因果核权重贡献；合并分子分母后统一归一化。
def linear_chunk(q,k,v,chunk=4,eps=1e-12):
    assert chunk>0
    q,k=phi(q),phi(k)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])); z=np.zeros(q.shape[:-2]+(q.shape[-1],))
    outs=[]
    for a in range(0,q.shape[-2],chunk):
        qc,kc,vc=q[...,a:a+chunk,:],k[...,a:a+chunk,:],v[...,a:a+chunk,:]
        # NumPy：tri(n) 得到含对角线的下三角 1/0 矩阵，表示可以读取自己及过去。
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        weights=(qc@kc.swapaxes(-1,-2))*np.tri(qc.shape[-2])
        numerator=qc@s+weights@vc
        # NumPy：einsum("...tr,...r->...t")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        denominator=np.einsum('...tr,...r->...t',qc,z)[...,None]+weights.sum(-1,keepdims=True)
        outs.append(numerator/(denominator+eps))
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        s=s+kc.swapaxes(-1,-2)@vc; z=z+kc.sum(-2)
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return np.concatenate(outs,-2),(s,z)

lr,ls=linear_recurrent(q,k,v)
check('linear dense recurrence',lr,linear_dense(q,k,v))
for c in [1,4,16]:
    lc,lstate=linear_chunk(q,k,v,c)
    check('linear chunk '+str(c),lc,lr); check('linear chunk state '+str(c),lstate[0],ls[0])
qa,ka=phi(q[0,0,:4]),phi(k[0,0,:4]); va=v[0,0,:4]
heatmaps([('Phi K.T',ka.T),('V',va),('S = K.T @ V',ka.T@va),('Phi Q',qa),('Q @ S',qa@(ka.T@va))],
         'Noncausal numerator: associativity removes N x N')
# NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
# NumPy：abs 对每个元素取绝对值，形状不变。
print('Linear 与 softmax 的最大差异（预期非零）:',np.max(np.abs(lr-o)))

def linear_noncausal(q,k,v,eps=1e-12):
    fq,fk=phi(q),phi(k)
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    state=fk.swapaxes(-1,-2)@v; z=fk.sum(-2)
    # NumPy：einsum("...tr,...r->...t")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    return (fq@state)/(np.einsum('...tr,...r->...t',fq,z)[...,None]+eps)

# 先保存每步前缀状态，反向倒序累积 ds,dz；较早写入会影响所有后续输出。
def linear_backward(q,k,v,go,eps=1e-12):
    fq,fk=phi(q),phi(k)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])); z=np.zeros(q.shape[:-2]+(q.shape[-1],))
    tape=[]
    for t in range(q.shape[-2]):
        s=s+fk[...,t,:,None]*v[...,t,None,:]; z=z+fk[...,t,:]
        tape.append((s,z))
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    ds=np.zeros_like(s); dz=np.zeros_like(z)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    for t in reversed(range(q.shape[-2])):
        s,z=tape[t]; qt=fq[...,t,:]; kt=fk[...,t,:]; vt=v[...,t,:]
        # NumPy：einsum("...r,...rv->...v")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        num=np.einsum('...r,...rv->...v',qt,s)
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        den=(qt*z).sum(-1,keepdims=True)+eps
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        dn=go[...,t,:]/den; dh=-(go[...,t,:]*num).sum(-1,keepdims=True)/(den*den)
        # NumPy：einsum("...rv,...v->...r")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dq[...,t,:]=np.einsum('...rv,...v->...r',s,dn)+dh*z
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        ds=ds+qt[..., :,None]*dn[...,None,:]; dz=dz+dh*qt
        # NumPy：einsum("...rv,...v->...r")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dk[...,t,:]=np.einsum('...rv,...v->...r',ds,vt)+dz
        # NumPy：einsum("...rv,...r->...v")：对 r 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dv[...,t,:]=np.einsum('...rv,...r->...v',ds,kt)
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：minimum 比较对应位置取较小值；例如先将指数输入限制为不大于 0，避免溢出。
    return dq*np.where(q>=0,1,np.exp(np.minimum(q,0))),dk*np.where(k>=0,1,np.exp(np.minimum(k,0))),dv
check('noncausal linear associativity',linear_noncausal(q,k,v),linear_dense(q,k,v,causal=False))
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
small=[rng.normal(size=(1,1,3,2)) for _ in range(3)]; upstream=rng.normal(size=(1,1,3,2))
lg=linear_backward(*small,upstream)
for j in range(3):
    def loss(a):
        args=small.copy(); args[j]=a
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        return np.sum(linear_recurrent(*args)[0]*upstream)
    check('Linear finite diff '+str(j),lg[j],finite_difference(loss,small[j]),atol=3e-8,rtol=2e-6)

# alpha 按 key 行遗忘，beta 控制残差写入；return_tape 保存反向所需的每步中间状态。
def delta_recurrent(q,k,v,beta,alpha=None,state=None,return_tape=False):
    assert q.shape==k.shape and q.shape[:-1]==v.shape[:-1]==beta.shape
    # NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
    if alpha is None: alpha=np.ones_like(k)
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]; tape=[]
    for t in range(q.shape[-2]):
        qt,kt,vt,bt,at=q[...,t,:],k[...,t,:],v[...,t,:],beta[...,t],alpha[...,t,:]
        previous=s
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        decayed=at[..., :,None]*s
        # NumPy：einsum("...k,...kv->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        prediction=np.einsum('...k,...kv->...v',kt,decayed)
        error=vt-prediction
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        s=decayed+(bt[...,None]*kt)[..., :,None]*error[...,None,:]
        # NumPy：einsum("...k,...kv->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        outs.append(np.einsum('...k,...kv->...v',qt,s))
        if return_tape: tape.append((previous,decayed,error,s))
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    result=(np.stack(outs,-2),s)
    return result+(tape,) if return_tape else result

qn,kn=normalize(q),normalize(k)
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
beta=sigmoid(rng.normal(size=(B,H,N)))
# NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
scalar_alpha=np.exp(-softplus(rng.normal(size=(B,H,N,1)))*.15)
# NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
vector_alpha=np.exp(-softplus(rng.normal(size=(B,H,N,DK)))*.15)
delta,_=delta_recurrent(qn,kn,v,beta)
gated,_=delta_recurrent(qn,kn,v,beta,scalar_alpha)
kda,kda_state=delta_recurrent(qn,kn,v,beta,vector_alpha)
# NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
check('scalar KDA reduces to GDN',delta_recurrent(qn,kn,v,beta,np.broadcast_to(scalar_alpha,kn.shape))[0],gated)
# NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
check('unit decay reduces to DeltaNet',delta_recurrent(qn,kn,v,beta,np.ones_like(kn))[0],delta)
# 一个两通道的数值例子：可以直接观察行衰减、残差与外积。
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
s0=np.array([[2.,1.],[3.,4.]])
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
kt=normalize(np.array([1.,2.])); vt=np.array([2.,-1.]); at=np.array([.9,.2]); bt=.8
sb=at[:,None]*s0; pred=kt@sb; err=vt-pred; update=bt*kt[:,None]*err[None,:]
# NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
heatmaps([('S previous',s0),('D @ S',sb),('pred ; target',np.stack([pred,vt])),
          ('beta k outer error',update),('S new',sb+update)], 'Decay -> predict -> residual -> rank-one write',cmap='coolwarm')
# beta=1 单位 key 的精确覆写。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
onek=normalize(rng.normal(size=(1,1,1,3))); onev=rng.normal(size=(1,1,1,2))
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
initial=rng.normal(size=(1,1,3,2))
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
check('unit key overwrite',delta_recurrent(onek,onek,onev,np.ones((1,1,1)),state=initial)[0],onev)

# 按行前代求解 (I+l)out=r；第 i 行只依赖前 i 行，i=0 时空和为零。
def unit_lower_solve(l,r):
    # 求 (I + 严格下三角 l) x = r；批量前代 O(C^2 * dv)。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    out=np.zeros_like(r)
    for i in range(r.shape[-2]):
        # NumPy：einsum("...j,...jv->...v")：对 j 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        out[...,i,:]=r[...,i,:]-np.einsum('...j,...jv->...v',l[...,i,:i],out[...,:i,:])
    return out

# 构造累计门、更新系数、读取系数和块尾 key；stable 路径允许门精确为零。
def chunk_factors(q,k,alpha,method):
    c,dk=k.shape[-2:]
    # NumPy：cumprod 沿指定轴做前缀乘积；axis=-2 是 token 轴，得到每个通道截至当前的累计门。
    g=np.cumprod(alpha,axis=-2)
    if method=='gemm':
        # NumPy：any 检查是否至少一个元素为真；用来拒绝数值不安全的累计门。
        if np.any(g<1e-100):
            raise ValueError('累计门过小或为零，请用 stable 路径或更小 chunk')
        kr=k/g
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        f=(k*g)@kr.swapaxes(-1,-2)
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        e=(q*g)@kr.swapaxes(-1,-2)
        kend=(g[...,-1:,:]/g)*k
    elif method=='stable':
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        f=np.zeros(k.shape[:-2]+(c,c)); e=np.zeros_like(f)
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        kend=np.zeros_like(k)
        for j in range(c):
            # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
            decay=np.ones(k.shape[:-2]+(dk,))
            for i in range(j,c):
                if i>j: decay=decay*alpha[...,i,:]
                weighted=decay*k[...,j,:]
                # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
                f[...,i,j]=np.sum(k[...,i,:]*weighted,-1)
                # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
                e[...,i,j]=np.sum(q[...,i,:]*weighted,-1)
            kend[...,j,:]=decay*k[...,j,:]
    else: raise ValueError(method)
    # NumPy：tril 保留最后两轴的下三角；k=-1 排除对角线，适合只依赖更早 token 的更新系数。
    return g, np.tril(f,-1),np.tril(e),kend

# 块内先解伪 value u，再读出与更新块尾；返回值必须与同参数的逐 token 递推一致。
def delta_chunk(q,k,v,beta,alpha=None,chunk=4,state=None,method='stable'):
    assert chunk>0
    # NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
    if alpha is None: alpha=np.ones_like(k)
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    s=np.zeros(q.shape[:-2]+(q.shape[-1],v.shape[-1])) if state is None else state.copy()
    outs=[]
    for start in range(0,q.shape[-2],chunk):
        sl=slice(start,start+chunk)
        qc,kc,vc,bc,ac=q[...,sl,:],k[...,sl,:],v[...,sl,:],beta[...,sl],alpha[...,sl,:]
        g,f,e,kend=chunk_factors(qc,kc,ac,method)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        l=bc[..., :,None]*f
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        rhs=bc[..., :,None]*(vc-(kc*g)@s)
        u=unit_lower_solve(l,rhs)
        outs.append((qc*g)@s+e@u)
        # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
        s=g[...,-1,:,None]*s+kend.swapaxes(-1,-2)@u
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return np.concatenate(outs,-2),s

for label,al in [('delta',None),('gdn',scalar_alpha),('kda',vector_alpha)]:
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    ini=rng.normal(size=(B,H,DK,DV))*.1
    ref,rs=delta_recurrent(qn,kn,v,beta,al,ini)
    for c in [1,4,16]:
        for method in ['stable','gemm']:
            actual,ss=delta_chunk(qn,kn,v,beta,al,c,ini,method)
            check(f'{label} chunk={c} {method}',actual,ref)
            check(f'{label} final state={c} {method}',ss,rs)
# 精确零门与极强衰减：稳定区间乘积路径。
az=vector_alpha.copy(); az[:,:,2,:]=0; az[:,:,5,:]=1e-200
check('zero/strong decay chunk',delta_chunk(qn,kn,v,beta,az,4)[0],delta_recurrent(qn,kn,v,beta,az)[0])
g,f,e,kend=chunk_factors(qn[:,:,:4],kn[:,:,:4],vector_alpha[:,:,:4],'stable')
heatmaps([('G: time x key',g[0,0]),('L: strict lower',(beta[:,:,:4,None]*f)[0,0]),
          ('E: causal read',e[0,0]),('K end',kend[0,0])], 'Chunk factors: solve (I+L) U = beta (V - Kg S0)')

# 保存所有前缀仿射变换作为正确性基准；不是推荐生产实现，矩阵连乘成本很高。
def delta_affine_scan(q,k,v,beta,alpha,state=None):
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：eye(dk) 生成单位矩阵，只有对角为 1；左乘状态保持原值。
    eye=np.eye(k.shape[-1])
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    aa=(eye-beta[...,None,None]*k[..., :,None]*k[...,None,:])*alpha[...,None,:]
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    bb=beta[...,None,None]*k[..., :,None]*v[...,None,:]
    n=q.shape[-2]; gap=1
    while gap<n:
        # 必须从旧数组读，不能在同一级中原地污染后续输入。
        na=aa.copy(); nb=bb.copy()
        na[...,gap:,:,:]=aa[...,gap:,:,:]@aa[...,:-gap,:,:]
        nb[...,gap:,:,:]=aa[...,gap:,:,:]@bb[...,:-gap,:,:]+bb[...,gap:,:,:]
        aa,bb=na,nb; gap*=2
    if state is not None: bb=bb+aa@state[...,None,:,:]
    # NumPy：einsum("...tk,...tkv->...tv")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
    return np.einsum('...tk,...tkv->...tv',q,bb),bb[...,-1,:,:]
check('affine scan oracle',delta_affine_scan(qn,kn,v,beta,vector_alpha)[0],kda)

# 按读出→外积写入→残差预测→行衰减的反序求导，h 带着未来状态梯度。
def delta_backward(q,k,v,beta,alpha,tape,go,final_grad=None):
    # NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
    alpha=np.broadcast_to(alpha,k.shape)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    db=np.zeros_like(beta); da=np.zeros_like(alpha)
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    h=np.zeros_like(tape[0][0]) if final_grad is None else final_grad.copy()
    for t in reversed(range(q.shape[-2])):
        prev,bar,err,st=tape[t]
        qt,kt,bt,at,gt=q[...,t,:],k[...,t,:],beta[...,t],alpha[...,t,:],go[...,t,:]
        # NumPy：einsum("...kv,...v->...k")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dq[...,t,:]=np.einsum('...kv,...v->...k',st,gt)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        h=h+qt[..., :,None]*gt[...,None,:]
        # NumPy：einsum("...kv,...k,...v->...")：对 k,v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        db[...,t]=np.einsum('...kv,...k,...v->...',h,kt,err)
        # NumPy：einsum("...kv,...v->...k")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        dk[...,t,:]=bt[...,None]*np.einsum('...kv,...v->...k',h,err)
        # NumPy：einsum("...kv,...k->...v")：对 k 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        de=bt[...,None]*np.einsum('...kv,...k->...v',h,kt)
        dv[...,t,:]=de
        # NumPy：einsum("...kv,...v->...k")：对 v 轴乘积求和；箭头右边规定输出轴顺序，... 保留共同前导轴。
        dk[...,t,:]-=np.einsum('...kv,...v->...k',bar,de)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        dbar=h-kt[..., :,None]*de[...,None,:]
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        da[...,t,:]=np.sum(dbar*prev,axis=-1)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        h=at[..., :,None]*dbar
    return dq,dk,dv,db,da,h

# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
args=[normalize(rng.normal(size=(1,1,3,2))),normalize(rng.normal(size=(1,1,3,2))),
      # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
      rng.normal(size=(1,1,3,2)),rng.uniform(.2,.8,(1,1,3)),rng.uniform(.2,.9,(1,1,3,2))]
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
ini=rng.normal(size=(1,1,2,2)); outs,ss,tape=delta_recurrent(*args,state=ini,return_tape=True)
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
gout=rng.normal(size=outs.shape); gs=rng.normal(size=ss.shape)
analytic=delta_backward(*args,tape,gout,gs)
for j in range(6):
    vals=args+[ini]
    def loss(z):
        aa=vals.copy(); aa[j]=z
        yy,st=delta_recurrent(*aa[:5],state=aa[5])
        # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
        return np.sum(yy*gout)+np.sum(st*gs)
    check('KDA backward finite diff '+str(j),analytic[j],finite_difference(loss,vals[j]),atol=3e-8,rtol=2e-6)
# 在输入 beta logits 上做一次真正的梯度下降，证明反向可用于优化。
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
logits=np.zeros((1,1,3)); target=rng.normal(size=outs.shape)
def beta_loss(b):
    yy,_=delta_recurrent(args[0],args[1],args[2],sigmoid(b),args[4],state=ini)
    # NumPy：sum 沿指定轴求和；keepdims 保留被归约轴，未指定 axis 则把所有元素加成标量。
    return .5*np.sum((yy-target)**2)
y,_,tp=delta_recurrent(args[0],args[1],args[2],sigmoid(logits),args[4],state=ini,return_tape=True)
db=delta_backward(args[0],args[1],args[2],sigmoid(logits),args[4],tp,y-target)[3]
new_logits=logits-.01*db*sigmoid(logits)*(1-sigmoid(logits))
require('one gradient step decreases loss', beta_loss(new_logits)<beta_loss(logits))
print('beta gate training loss:',beta_loss(logits),'->',beta_loss(new_logits))

# m/l 每条 query 一个数，a 每条 query 一个 value 向量；新最大值改变时一起重标定。
def online_update(m,l,a,x,v):
    # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    new_m=np.maximum(m,np.max(x,axis=-1))
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    safe=np.where(np.isfinite(new_m),new_m,0.)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    rescale=np.exp(m-safe)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    weights=np.exp(x-safe[...,None])
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    new_l=rescale*l+weights.sum(-1)
    # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
    new_a=rescale[...,None]*a+weights@v
    return new_m,new_l,new_a

# 只计算当前 Q/K 小块，并按全局位置遮蔽未来；外部 keep 同时切取对应行列。
def tile_scores(qc,kc,i,j,causal,keep,q_start=0):
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    scores=qc@kc.swapaxes(-1,-2)/math.sqrt(qc.shape[-1])
    if causal:
        mask=position_mask(qc.shape[-2],kc.shape[-2],i+q_start,j)
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        scores=np.where(mask,scores,-np.inf)
    if keep is not None:
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        scores=np.where(keep[...,i:i+qc.shape[-2],j:j+kc.shape[-2]],scores,-np.inf)
    return scores

# K/V tile 外循环、query tile 内循环；保留归一化输出，在更新时恢复未归一化分子。
def flash1_model(q,k,v,rows=4,cols=3,causal=True,keep=None,q_start=0):
    assert rows>0 and cols>0
    # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    m=np.full(q.shape[:-1],-np.inf); l=np.zeros_like(m)
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    out=np.zeros(q.shape[:-1]+(v.shape[-1],))
    for j in range(0,k.shape[-2],cols):
        kc,vc=k[...,j:j+cols,:],v[...,j:j+cols,:]
        for i in range(0,q.shape[-2],rows):
            sl=slice(i,i+rows); qc=q[...,sl,:]
            x=tile_scores(qc,kc,i,j,causal,keep,q_start)
            mm,ll,aa=online_update(m[...,sl],l[...,sl],out[...,sl,:]*l[...,sl,None],x,vc)
            # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
            # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
            # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
            out[...,sl,:]=np.divide(aa,ll[...,None],out=np.zeros_like(aa),where=ll[...,None]>0)
            m[...,sl]=mm; l[...,sl]=ll
    return out

# 每个 query tile 拥有一组 m,l,a；依次消费 KV tiles，末尾归一化一次。
def flash2_model(q,k,v,rows=4,cols=3,causal=True,keep=None,q_start=0,return_lse=False):
    assert rows>0 and cols>0
    outputs=[]; logsum=[]
    for i in range(0,q.shape[-2],rows):
        qc=q[...,i:i+rows,:]
        # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        m=np.full(qc.shape[:-1],-np.inf); l=np.zeros_like(m)
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        a=np.zeros(qc.shape[:-1]+(v.shape[-1],))
        for j in range(0,k.shape[-2],cols):
            kc,vc=k[...,j:j+cols,:],v[...,j:j+cols,:]
            x=tile_scores(qc,kc,i,j,causal,keep,q_start)
            m,l,a=online_update(m,l,a,x,vc)
        # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        outputs.append(np.divide(a,l[...,None],out=np.zeros_like(a),where=l[...,None]>0))
        # NumPy：log 是自然对数；调用前须保证合法正输入，空行另设安全值。
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        logsum.append(m+np.log(np.where(l>0,l,1.)))
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    result=np.concatenate(outputs,-2)
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return (result,np.concatenate(logsum,-1)) if return_lse else result

for causal in [False,True]:
    keep=rng.random((B,1,N,N))>.25; keep[:,:,0,:]=False
    ref,_=softmax_attention(q,k,v,causal=causal,keep=keep)
    for r,c in [(1,1),(4,3),(16,7)]:
        for fn in [flash1_model,flash2_model]:
            check(f'{fn.__name__} {causal} {r}x{c}',fn(q,k,v,r,c,causal,keep),ref)
# 大 logits 测试；不是先 exp(qk) 再 softmax。
check('flash stable large logits',flash2_model(q*300,k*300,v),softmax_attention(q*300,k*300,v,causal=True)[0])
check('flash decode offset',flash2_model(q[:,:,-1:],k,v,q_start=N-1),o[:,:,-1:])

from matplotlib.patches import Rectangle
fig,axes=plt.subplots(1,2,figsize=(12,4))
# NumPy：tri(n) 得到含对角线的下三角 1/0 矩阵，表示可以读取自己及过去。
mask=np.tri(9); axes[0].imshow(mask,cmap='Blues',vmin=0,vmax=1)
for i in range(0,9,3):
    for j in range(0,9,4):
        axes[0].add_patch(Rectangle((j-.5,i-.5),min(4,9-j),3,fill=False,edgecolor='#d57525',linewidth=2))
for i in range(9):
    for j in range(9):
        axes[0].text(j,i,f'{i},{j}' if j<=i else 'x',fontsize=7,ha='center',va='center')
axes[0].set_title('Causal score tiles: R=3, T=4'); axes[0].set_xlabel('key position j'); axes[0].set_ylabel('query position i')
# 选后3行，确保多个 KV tile 都有真实贡献。
# NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
qt=q[0,0,6:9]; mm=np.full(3,-np.inf); ll=np.zeros(3); aa=np.zeros((3,DV)); history=[]
for j in range(0,9,4):
    scores=tile_scores(qt,k[0,0,j:j+4],6,j,True,None)
    mm,ll,aa=online_update(mm,ll,aa,scores,v[0,0,j:j+4])
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    history.append(np.stack([mm,ll,aa[:,0]],-1))
# NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
history=np.concatenate(history,axis=0)
axes[1].imshow(history,cmap='coolwarm',aspect='auto')
axes[1].set_xticks([0,1,2],['m','l','a[value=0]'])
axes[1].set_yticks(range(9),[f'tile {j}, query {i}' for j in range(3) for i in range(6,9)])
# NumPy：ndenumerate 同时返回坐标元组与该格数值，用来为热力图加文字。
for (i,j),val in np.ndenumerate(history): axes[1].text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=8)
axes[1].set_title('Running online-softmax statistics'); plt.tight_layout(); plt.show()
check('illustrated tile output',aa/ll[:,None],o[0,0,6:9])

# 由 score 与 LSE 重算局部概率，再累加 Q/K/V 梯度；这里使用单线程确定性累加。
def flash_backward(q,k,v,go,out,lse,rows=4,cols=3,causal=True,keep=None):
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    dq=np.zeros_like(q); dk=np.zeros_like(k); dv=np.zeros_like(v)
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    d=(go*out).sum(-1); scale=q.shape[-1]**-.5
    for i in range(0,q.shape[-2],rows):
        qi=q[...,i:i+rows,:]; gi=go[...,i:i+rows,:]
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
        ll=lse[...,i:i+rows]; safe=np.where(np.isfinite(ll),ll,0.)
        for j in range(0,k.shape[-2],cols):
            kj,vj=k[...,j:j+cols,:],v[...,j:j+cols,:]
            x=tile_scores(qi,kj,i,j,causal,keep)
            # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
            # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
            pp=np.exp(x-safe[...,None])
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            dp=gi@vj.swapaxes(-1,-2)
            ds=pp*(dp-d[...,i:i+rows,None])
            dq[...,i:i+rows,:]+=scale*(ds@kj)
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            dk[...,j:j+cols,:]+=scale*(ds.swapaxes(-1,-2)@qi)
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            dv[...,j:j+cols,:]+=pp.swapaxes(-1,-2)@gi
    return dq,dk,dv
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
fo,fl=flash2_model(q,k,v,return_lse=True); gg=rng.normal(size=fo.shape)
fg=flash_backward(q,k,v,gg,fo,fl)
sg=softmax_backward(q,k,v,p,gg)
for j in range(3): check('Flash recomputed backward '+str(j),fg[j],sg[j])

# 用双槽位模拟加载/消费依赖；实际仍是顺序 CPU 执行，不含 CUDA 异步指令。
def flash3_pipeline_model(q,k,v,rows=4,cols=3,causal=True):
    outputs=[]; trace=[]
    for i in range(0,q.shape[-2],rows):
        # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
        qc=q[...,i:i+rows,:]; m=np.full(qc.shape[:-1],-np.inf)
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        l=np.zeros_like(m); a=np.zeros(qc.shape[:-1]+(v.shape[-1],))
        buffers=[None,None]; starts=list(range(0,k.shape[-2],cols))
        def load(step):
            slot=step%2; j=starts[step]
            assert buffers[slot] is None
            buffers[slot]=(j,k[...,j:j+cols,:].copy(),v[...,j:j+cols,:].copy())
            trace.append((i,step,'load',slot))
        load(0)
        for step in range(len(starts)):
            if step+1<len(starts): load(step+1)
            slot=step%2; j,kc,vc=buffers[slot]
            x=tile_scores(qc,kc,i,j,causal,None)
            m,l,a=online_update(m,l,a,x,vc)
            trace.append((i,step,'consume',slot)); buffers[slot]=None
        # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        outputs.append(np.divide(a,l[...,None],out=np.zeros_like(a),where=l[...,None]>0))
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    return np.concatenate(outputs,-2),trace
f3,trace=flash3_pipeline_model(q,k,v)
check('FA3 pipeline math',f3,o)
print('Double-buffer event trace:',trace[:8])
# 自画调度图：横轴为示意时间，非测量时间，不暗示精确 stall 数。
fig,ax=plt.subplots(figsize=(10,3))
loads=[0.,1.,3.,5.]
qk=[1.,2.,4.,6.]; sm=[2.,3.,5.,7.]; pv=[3.,5.,7.,8.]
for j in range(4):
    ax.broken_barh([(loads[j],.7)],(2.1,.6),facecolors='#5896c7')
    ax.broken_barh([(qk[j],1.)],(1.1,.6),facecolors='#edb14c')
    ax.broken_barh([(pv[j],1.)],(1.1,.6),facecolors='#d98749')
    ax.broken_barh([(sm[j],.7)],(.1,.6),facecolors='#5cad8b')
    ax.text(loads[j]+.05,2.4,f'L{j}',fontsize=9)
    ax.text(qk[j]+.05,1.4,f'QK{j}',fontsize=9)
    ax.text(pv[j]+.05,1.4,f'PV{j}',fontsize=9)
    ax.text(sm[j]+.05,.4,f'SM{j}',fontsize=9)
ax.set_yticks([.4,1.4,2.4],['Softmax','Tensor Core work','TMA producer'])
ax.set_xlabel('Illustrative dependency-respecting schedule, not measured cycles')
ax.set_xlim(-.2,9.3); ax.set_ylim(0,3); plt.tight_layout(); plt.show()

def hadamard(n):
    assert n>0 and n&(n-1)==0
    # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
    h=np.ones((1,1))
    # NumPy：block 按嵌套列表拼出大矩阵；这里构造 Hadamard 的四个正负子块。
    while h.shape[0]<n: h=np.block([[h,h],[h,-h]])
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    return h/np.sqrt(n)
def quantize_uniform(x):
    # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    # NumPy：abs 对每个元素取绝对值，形状不变。
    scale=np.maximum(np.max(np.abs(x),axis=(-2,-1),keepdims=True),1e-12)/127
    # NumPy：clip 把每格限制在给定上下界之间，用来模拟有限的量化范围。
    # NumPy：round 将每格舍入到邻近整数值；返回数组通常仍是浮点 dtype。
    return np.clip(np.round(x/scale),-127,127)*scale
rr=hadamard(DK)*rng.choice([-1.,1.],size=(1,DK))
# NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
check('orthogonal transform invariant logits',(q@rr)@(k@rr).swapaxes(-1,-2),q@k.swapaxes(-1,-2))
# NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
original=q@k.swapaxes(-1,-2)
for label,qa,ka in [('plain',q,k),('rotated',q@rr,k@rr)]:
    # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
    # NumPy：linalg.norm 沿给定轴求欧氏长度 sqrt(sum(x*x))，不是对矩阵每格取绝对值。
    err=np.linalg.norm(quantize_uniform(qa)@quantize_uniform(ka).swapaxes(-1,-2)-original)/np.linalg.norm(original)
    print('Uniform INT8-style demo relative logits error',label,err)

class PagePool:
    def __init__(self,pages,heads,page_size,dk,dv):
        assert pages>0 and page_size>0
        self.P=page_size; self.H=heads
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        self.k=np.zeros((pages,heads,page_size,dk))
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        self.v=np.zeros((pages,heads,page_size,dv))
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        self.refs=np.zeros(pages,dtype=int)
        self.free=list(reversed(range(pages)))
    def alloc(self):
        if not self.free: raise MemoryError('KV page pool exhausted')
        p=self.free.pop(); assert self.refs[p]==0
        self.refs[p]=1
        self.k[p].fill(0); self.v[p].fill(0)
        return p
    def retain(self,p):
        assert self.refs[p]>0; self.refs[p]+=1
    def release(self,p):
        assert self.refs[p]>0; self.refs[p]-=1
        if self.refs[p]==0: self.free.append(p)
    def audit(self):
        assert len(self.free)==len(set(self.free))
        # NumPy：flatnonzero 返回扁平数组中非零元素的整数位置，用来核对空闲页号。
        assert set(self.free)==set(np.flatnonzero(self.refs==0))
        # NumPy：all 检查布尔数组是否全部为真，返回真/假而非数值误差大小。
        assert np.all(self.refs>=0)

class PagedSequence:
    def __init__(self,pool):
        self.pool=pool; self.table=[]; self.length=0; self.closed=False
    def append(self,k,v):
        assert not self.closed
        p=self.pool
        assert k.shape==(p.H,p.k.shape[-1]) and v.shape==(p.H,p.v.shape[-1])
        offset=self.length%p.P
        if offset==0:
            physical=p.alloc(); self.table.append(physical)
        else:
            physical=self.table[-1]
            if p.refs[physical]>1:
                new=p.alloc()  # 失败时，旧页及引用计数还没有改动。
                p.k[new]=p.k[physical]; p.v[new]=p.v[physical]
                p.release(physical); self.table[-1]=new; physical=new
        p.k[physical,:,offset,:]=k; p.v[physical,:,offset,:]=v
        self.length+=1
    def fork(self):
        assert not self.closed
        child=PagedSequence(self.pool); child.table=self.table.copy(); child.length=self.length
        for p in child.table: self.pool.retain(p)
        return child
    def blocks(self):
        assert not self.closed
        for logical,physical in enumerate(self.table):
            count=min(self.pool.P,self.length-logical*self.pool.P)
            yield self.pool.k[physical,:,:count], self.pool.v[physical,:,:count]
    def gather(self):
        blocks=list(self.blocks())
        if not blocks:
            # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
            return np.zeros((self.pool.H,0,self.pool.k.shape[-1])),np.zeros((self.pool.H,0,self.pool.v.shape[-1]))
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        return np.concatenate([x[0] for x in blocks],1),np.concatenate([x[1] for x in blocks],1)
    def decode(self,q):
        assert q.shape==(self.pool.H,self.pool.k.shape[-1]) and self.length>0
        # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
        # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
        m=np.full((self.pool.H,1),-np.inf); l=np.zeros_like(m)
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        a=np.zeros((self.pool.H,1,self.pool.v.shape[-1]))
        for kk,vv in self.blocks():
            # NumPy：swapaxes(-1,-2) 只交换最后两条矩阵轴；batch/head 轴保持原位。
            scores=q[:,None,:]@kk.swapaxes(-1,-2)/math.sqrt(q.shape[-1])
            m,l,a=online_update(m,l,a,scores,vv)
        # NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
        return (a/l[...,None])[:,0,:]
    def close(self):
        if self.closed: return
        for p in self.table: self.pool.release(p)
        self.table=[]; self.length=0; self.closed=True

pool=PagePool(12,H,3,DK,DV)
# 先占用一页，构造非连续的物理页分布。
hole=pool.alloc(); seq=PagedSequence(pool)
for t in range(4): seq.append(k[0,:,t],v[0,:,t])
pool.release(hole)
child=seq.fork(); old=seq.gather()
child.append(k[0,:,4],v[0,:,4])
check('COW parent unchanged K',seq.gather()[0],old[0]); check('COW parent unchanged V',seq.gather()[1],old[1])
require('partial last page copied',seq.table[-1]!=child.table[-1])
for t in range(5,N): child.append(k[0,:,t],v[0,:,t])
kk,vv=child.gather()
check('paged gather logical order',kk,k[0])
check('paged decode dense',child.decode(q[0,:,-1]),softmax_attention(q[0,:,-1:, :],kk,vv)[0][:,0])
pool.audit()
fig,ax=plt.subplots(figsize=(9,3))
for i,pid in enumerate(child.table):
    ax.text(i*2+.5,2,f'Logical {i}',ha='center',bbox=dict(boxstyle='round',fc='#e6f1fa'))
    ax.text(i*2+.5,.4,f'Physical {pid}',ha='center',bbox=dict(boxstyle='round',fc='#e7f4eb'))
    ax.annotate('',xy=(i*2+.5,.75),xytext=(i*2+.5,1.8),arrowprops=dict(arrowstyle='->'))
ax.set_xlim(-.5,7); ax.set_ylim(0,2.5); ax.axis('off'); ax.set_title('Page table: address mapping, each block has 3 token slots'); plt.show()
seq.close(); child.close(); pool.audit()
require('all pages released',len(pool.free)==12)
# 满页共享后追加：旧页保持共享，新建下一页。
p2=PagePool(3,1,2,2,2); a=PagedSequence(p2)
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
for _ in range(2): a.append(np.ones((1,2)),np.ones((1,2)))
# NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
b=a.fork(); b.append(np.zeros((1,2)),np.zeros((1,2)))
require('full-page fork no old-page copy',a.table[0]==b.table[0] and len(b.table)==2)
a.close(); b.close(); p2.audit()
# OOM 不损坏原始共享尾页。
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
p3=PagePool(1,1,2,2,2); a=PagedSequence(p3); a.append(np.ones((1,2)),np.ones((1,2))); b=a.fork()
try:
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    b.append(np.zeros((1,2)),np.zeros((1,2)))
    raise AssertionError('Expected OOM')
except MemoryError: pass
require('OOM leaves references intact',p3.refs[0]==2 and b.length==1)
a.close(); b.close(); p3.audit()

@dataclass(eq=False)
class RadixNode:
    edge: tuple=()
    kv: object=None  # (K,V)，各为 [tokens,heads,dim]
    parent: object=None
    children: dict=field(default_factory=dict)
    pins: int=0
    stamp: int=0

def lcp(a,b):
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]: i+=1
    return i

class RadixCache:
    def __init__(self,namespace):
        self.namespace=namespace; self.root=RadixNode(); self.clock=0
    def tick(self,node):
        self.clock+=1; node.stamp=self.clock
    def insert(self,tokens,k,v,namespace):
        assert namespace==self.namespace, 'cache namespace mismatch'
        tokens=tuple(tokens); assert len(tokens)==len(k)==len(v)
        node=self.root; pos=0
        while pos<len(tokens):
            rem=tokens[pos:]; child=node.children.get(rem[0])
            if child is None:
                new=RadixNode(rem,(k[pos:].copy(),v[pos:].copy()),node)
                node.children[rem[0]]=new; self.tick(new); return
            n=lcp(rem,child.edge)
            # 若相同 prefix 的 KV 不同，拒绝静默命中。
            # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
            np.testing.assert_allclose(child.kv[0][:n],k[pos:pos+n],atol=1e-10)
            # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
            np.testing.assert_allclose(child.kv[1][:n],v[pos:pos+n],atol=1e-10)
            if n<len(child.edge):
                mid=RadixNode(child.edge[:n],(child.kv[0][:n].copy(),child.kv[1][:n].copy()),node)
                mid.pins=child.pins; mid.stamp=child.stamp
                node.children[rem[0]]=mid
                child.edge=child.edge[n:]; child.kv=(child.kv[0][n:].copy(),child.kv[1][n:].copy())
                child.parent=mid; mid.children[child.edge[0]]=child
                child=mid
            self.tick(child); node=child; pos+=n
    def match(self,tokens,namespace):
        assert namespace==self.namespace, 'cache namespace mismatch'
        tokens=tuple(tokens); node=self.root; pos=0; ks=[]; vs=[]
        while pos<len(tokens):
            child=node.children.get(tokens[pos])
            if child is None: break
            n=lcp(tokens[pos:],child.edge)
            if not n: break
            ks.append(child.kv[0][:n]); vs.append(child.kv[1][:n])
            pos+=n; node=child; self.tick(node)
            if n<len(child.edge): break
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        return pos,(np.concatenate(ks) if ks else None,np.concatenate(vs) if vs else None),node
    @contextmanager
    def lease(self,tokens,namespace):
        matched=self.match(tokens,namespace); node=matched[2]
        cur=node
        while cur is not self.root: cur.pins+=1; cur=cur.parent
        try: yield matched
        finally:
            # 动态沿当前父指针回溯，支持 lease 期间边分裂。
            cur=node
            while cur is not self.root:
                cur.pins-=1; assert cur.pins>=0; cur=cur.parent
    def nodes(self):
        stack=list(self.root.children.values()); result=[]
        while stack:
            n=stack.pop(); result.append(n); stack.extend(n.children.values())
        return result
    @property
    def tokens(self): return sum(len(n.edge) for n in self.nodes())
    def evict_to(self,budget):
        assert budget>=0
        while self.tokens>budget:
            leaves=[n for n in self.nodes() if not n.children and n.pins==0]
            if not leaves: break
            victim=min(leaves,key=lambda n:n.stamp)
            del victim.parent.children[victim.edge[0]]
        return self.tokens<=budget
    def edges(self):
        return sorted([(n.edge,n.pins,len(n.children)) for n in self.nodes()])

ns=('toy-model-v1','tokenizer-v1','position-start-0','no-adapter')
cache=RadixCache(ns)
# 数据只为结构测试；真实 KV 必须包含上下文影响，下一单元会用 causal 层验证。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
embk=rng.normal(size=(20,H,DK)); embv=rng.normal(size=(20,H,DV))
def insert_ids(ids): cache.insert(ids,embk[ids],embv[ids],ns)
insert_ids([1,2,3,4]); insert_ids([1,2,5]); insert_ids([8,9])
require('radix unique prefix storage',cache.tokens==7)
matched,kv,node=cache.match([1,2,3,9],ns)
require('partial compressed-edge match',matched==3)
check('radix KV match order',kv[0],embk[[1,2,3]])
with cache.lease([1,2,3,4],ns):
    require('pinned cache cannot evict all',not cache.evict_to(0))
    require('active prefix survives',cache.match([1,2,3,4],ns)[0]==4)
require('unpin enables eviction',cache.evict_to(0) and cache.tokens==0)
insert_ids([1,2,3,4])
with cache.lease([1,2,3,4],ns):
    insert_ids([1,2,5])
    require('split inherits path pin',any(n.edge==(1,2) and n.pins==1 for n in cache.nodes()))
require('split unpin balanced',all(n.pins==0 for n in cache.nodes()))
try:
    cache.match([1,2],('different-model',))
    raise AssertionError('namespace should fail')
except AssertionError as e:
    assert str(e)=='cache namespace mismatch'
    TESTS.append('namespace isolation')
print('Compressed edges (tokens, pins, children):',cache.edges())

dmodel=8; hh=2; dd=4
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
embedding=rng.normal(size=(30,dmodel))*.2
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
weights=[tuple(rng.normal(size=(dmodel,dmodel))*.2 for _ in range(4)) for _ in range(2)]
def toy_layer(x,w,prefix=None):
    wq,wk,wv,wo=w
    qq=split_heads((x@wq)[None],hh); kk=split_heads((x@wk)[None],hh); vv=split_heads((x@wv)[None],hh)
    length=0 if prefix is None else prefix[0].shape[0]
    if prefix is None: allk,allv=kk,vv
    else:
        pk,pv=prefix
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
        allk=np.concatenate([pk.transpose(1,0,2)[None],kk],axis=2)
        # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
        # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
        allv=np.concatenate([pv.transpose(1,0,2)[None],vv],axis=2)
    yy=softmax_attention(qq,allk,allv,causal=True,q_start=length)[0]
    result=x+merge_heads(yy)[0]@wo
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    return result,(allk[0].transpose(1,0,2),allv[0].transpose(1,0,2))

def full_toy(ids):
    x=embedding[ids]; states=[]
    for w in weights:
        x,kv=toy_layer(x,w); states.append(kv)
    return x,states
prefix=[2,4,6,8]; ids=prefix+[10,12,14]
_,states=full_toy(prefix)
layer_caches=[RadixCache(ns+('layer',i)) for i in range(2)]
for i,c in enumerate(layer_caches): c.insert(prefix,*states[i],ns+('layer',i))
x=embedding[ids[len(prefix):]]
for i,(w,c) in enumerate(zip(weights,layer_caches)):
    hit,cached,_=c.match(ids,ns+('layer',i))
    require('layer prefix hit '+str(i),hit==len(prefix))
    x,newkv=toy_layer(x,w,cached)
    c.insert(ids,*newkv,ns+('layer',i))
full,_=full_toy(ids)
check('two-layer radix cached suffix equals full recomputation',x,full[len(prefix):])
print('Cached prefix:',len(prefix),'new tokens:',len(ids)-len(prefix))
# 真正的压缩 radix 树图，边标签来自插入的 token IDs。
fig,ax=plt.subplots(figsize=(8,3))
positions={'root':(0,2),'prefix':(0,1),'left':(-1.7,0),'right':(1.7,0)}
for a,b,lab in [('root','prefix','[1, 2]'),('prefix','left','[3, 4]'),('prefix','right','[5]')]:
    x1,y1=positions[a]; x2,y2=positions[b]
    ax.annotate('',xy=(x2,y2+.15),xytext=(x1,y1-.1),arrowprops=dict(arrowstyle='->'))
    ax.text((x1+x2)/2+.1,(y1+y2)/2,lab,bbox=dict(fc='white',ec='none'))
for name,(xx,yy) in positions.items(): ax.text(xx,yy,name,ha='center',bbox=dict(boxstyle='round',fc='#e8f2fa'))
ax.set_xlim(-2.8,2.8); ax.set_ylim(-.4,2.5); ax.axis('off'); ax.set_title('Compressed radix tree: tokens on edges'); plt.show()

# 每个特征通道独立卷积，lag=0 是当前 token，lag>0 只读取更早 token。
def short_conv(x,w):
    # x [B,N,D], w [window,D], w[0] 是当前 token 的权重。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    out=np.zeros_like(x)
    for lag in range(min(len(w),x.shape[1])):
        out[:,lag:,:]+=x[:,:x.shape[1]-lag,:]*w[lag]
    return out

def silu(x): return x*sigmoid(x)

def make_kda_weights(dmodel,heads,dk,dv,rank=4,window=3,seed=7):
    rr=np.random.default_rng(seed)
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    def w(a,b): return rr.normal(size=(a,b))/np.sqrt(a)
    return dict(wq=w(dmodel,heads*dk),wk=w(dmodel,heads*dk),wv=w(dmodel,heads*dv),
                cq=w(window,heads*dk),ck=w(window,heads*dk),cv=w(window,heads*dv),
                ad=w(dmodel,rank),au=w(rank,heads*dk),bl=w(dmodel,heads),
                gd=w(dmodel,rank),gu=w(rank,heads*dv),wo=w(heads*dv,dmodel),
                # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
                # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
                alog=np.full((1,heads,1,dk),-2.),dtbias=np.zeros((1,heads,1,dk)),
                # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
                norm_weight=np.ones((1,heads,1,dv)))

# 投影→短因果卷积→激活/归一化→门→状态更新→归一化/输出门→合头投影。
def kda_layer(x,w,heads,method='recurrent'):
    qq=normalize(split_heads(silu(short_conv(x@w['wq'],w['cq'])),heads))
    kk=normalize(split_heads(silu(short_conv(x@w['wk'],w['ck'])),heads))
    vv=split_heads(silu(short_conv(x@w['wv'],w['cv'])),heads)
    raw=split_heads((x@w['ad'])@w['au'],heads)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    log_alpha=-np.exp(w['alog'])*softplus(raw+w['dtbias'])
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    alpha=np.exp(log_alpha)
    # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
    beta=sigmoid(x@w['bl']).transpose(0,2,1)
    if method=='recurrent': yy,_=delta_recurrent(qq,kk,vv,beta,alpha)
    elif method=='chunk': yy,_=delta_chunk(qq,kk,vv,beta,alpha,chunk=4)
    else: raise ValueError(method)
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    # NumPy：mean 沿指定轴取平均；这里对每个 token 的 value 通道计算均方。
    yy=yy/np.sqrt(np.mean(yy*yy,axis=-1,keepdims=True)+1e-6)*w['norm_weight']
    gate=sigmoid(split_heads((x@w['gd'])@w['gu'],heads))
    return merge_heads(yy*gate)@w['wo']

# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
x=rng.normal(size=(2,9,8)); kw=make_kda_weights(8,2,4,3)
y=kda_layer(x,kw,2); yc=kda_layer(x,kw,2,'chunk')
check('complete KDA layer chunk recurrence',yc,y,atol=1e-9)
x_future=x.copy(); x_future[:,5:]+=20
check('complete KDA layer causal',kda_layer(x_future,kw,2)[:,:5],y[:,:5])
print('KDA layer shape:',x.shape,'->',y.shape)

# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
lengths=np.array([32,64,128,256,512,1024,2048,4096])
kv_bytes=lengths*8*256*2
# NumPy：full_like 复制参照数组的形状和 dtype，再用给定标量填充。
state_bytes=np.full_like(lengths,8*128*128*4)
score_bytes=lengths**2*8*4
fig,ax=plt.subplots(figsize=(8,4))
for label,yy in [('FP16 KV: 8 heads',kv_bytes),('FP32 KDA state',state_bytes),('FP32 dense scores',score_bytes)]:
    ax.loglog(lengths,yy/2**20,marker='o',label=label)
ax.set_xlabel('Sequence length N'); ax.set_ylabel('MiB, single layer/request')
ax.set_title('Analytical storage only: inputs, weights and allocator excluded')
ax.grid(alpha=.2); ax.legend(); plt.tight_layout(); plt.show()

def median_ms(fn,repeats=3):
    fn(); samples=[]
    for _ in range(repeats):
        start=time.perf_counter(); fn(); samples.append((time.perf_counter()-start)*1000)
    # NumPy：median 取多次测量的中位数；它描述本机 CPU 计时，不是 GPU 吞吐。
    return float(np.median(samples))
bench=[]
for nn in [32,64,128]:
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    qq=normalize(rng.normal(size=(1,1,nn,8))); kk=normalize(rng.normal(size=qq.shape))
    # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
    # NumPy：full_like 复制参照数组的形状和 dtype，再用给定标量填充。
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    vv=rng.normal(size=(1,1,nn,8)); bb=np.full((1,1,nn),.7); aa=np.full_like(kk,.95)
    funcs={'Dense softmax':lambda:softmax_attention(qq,kk,vv,causal=True)[0],
           'FA2 math model':lambda:flash2_model(qq,kk,vv,16,16),
           'Linear recurrent':lambda:linear_recurrent(qq,kk,vv)[0],
           'KDA recurrent':lambda:delta_recurrent(qq,kk,vv,bb,aa)[0],
           'KDA GEMM chunk':lambda:delta_chunk(qq,kk,vv,bb,aa,16,method='gemm')[0]}
    for name,fn in funcs.items(): bench.append((nn,name,median_ms(fn)))
print('N    implementation            median CPU ms')
for nn,name,ms in bench: print(f'{nn:<4} {name:<25} {ms:9.4f}')
fig,ax=plt.subplots(figsize=(9,4))
for name in funcs:
    points=[(n,t) for n,label,t in bench if label==name]
    ax.plot([x[0] for x in points],[x[1] for x in points],marker='o',label=name)
ax.set_xlabel('N'); ax.set_ylabel('median CPU milliseconds'); ax.set_title('NumPy reference implementations, not GPU kernel benchmarks')
ax.legend(fontsize=8); ax.grid(alpha=.2); plt.tight_layout(); plt.show()

# KDA / GDN / DeltaNet 真正跨调用传递状态，而非每次从零开始。
# NumPy：ones_like 生成同形全 1；门取 1 表示不遗忘，而不是让状态本身变成 1。
# NumPy：broadcast_to 将大小为 1 的轴按目标形状广播，通常返回只读视图，不会自行推断轴含义。
for label,al in [('Delta',np.ones_like(kn)),('GDN',np.broadcast_to(scalar_alpha,kn.shape)),('KDA',vector_alpha)]:
    whole,final=delta_recurrent(qn,kn,v,beta,al)
    left,state=delta_recurrent(qn[:,:,:4],kn[:,:,:4],v[:,:,:4],beta[:,:,:4],al[:,:,:4])
    right,state=delta_recurrent(qn[:,:,4:],kn[:,:,4:],v[:,:,4:],beta[:,:,4:],al[:,:,4:],state)
    # NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
    check(label+' streaming split',np.concatenate([left,right],2),whole)
    check(label+' streaming final',state,final)
left,state=linear_recurrent(q[:,:,:4],k[:,:,:4],v[:,:,:4])
right,state=linear_recurrent(q[:,:,4:],k[:,:,4:],v[:,:,4:],state)
# NumPy：concatenate 沿已有轴接长数组；axis=-2 接 token，axis=-1 接通道，其他轴必须一致。
check('Linear streaming split',np.concatenate([left,right],2),lr)
# beta=0 时只有通道遗忘；从非零 S0 验证，避免零状态平凡通过。
# NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
sinit=rng.normal(size=(B,H,DK,DV)); bzero=np.zeros_like(beta)
_,sf=delta_recurrent(qn,kn,v,bzero,vector_alpha,sinit)
# NumPy：prod 沿指定轴把元素相乘；这里将多步遗忘门合成为总衰减。
# NumPy：None 插入长度为 1 的轴，使逐行缩放或外积通过广播完成；... 代表前面的所有轴。
check('beta zero preserves pure decay',sf,np.prod(vector_alpha,axis=2)[..., :,None]*sinit)
# 不同 query/KV 长度和任意 mask 的 Flash 对照。
qr=q[:,:,:5]; kr=k[:,:,:7]; vr=v[:,:,:7]
keep=rng.random((B,1,5,7))>.3; keep[:,:,2,:]=False
for fn in [flash1_model,flash2_model]:
    check(fn.__name__+' rectangular',fn(qr,kr,vr,3,4,False,keep),softmax_attention(qr,kr,vr,keep=keep)[0])
# MQA/GQA 手动分组 oracle。
# NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
gq=rng.normal(size=(1,4,5,3)); gk=rng.normal(size=(1,2,5,3)); gv=rng.normal(size=(1,2,5,2))
gout=gqa(gq,gk,gv,causal=True)
for head in range(4):
    check('GQA group '+str(head),gout[:,head:head+1],softmax_attention(gq[:,head:head+1],gk[:,head//2:head//2+1],gv[:,head//2:head//2+1],causal=True)[0])
# 因果性：未来值的巨大扰动不改变前5个输出。
vfuture=v.copy(); vfuture[:,:,5:]+=100
for name,fn in [('Softmax',lambda vv:softmax_attention(q,k,vv,causal=True)[0]),
                ('Linear',lambda vv:linear_recurrent(q,k,vv)[0]),
                ('KDA',lambda vv:delta_chunk(qn,kn,vv,beta,vector_alpha,4)[0]),
                ('Flash',lambda vv:flash2_model(q,k,vv))]:
    check(name+' no future leakage',fn(vfuture)[:,:,:5],fn(v)[:,:,:5])
# 覆写与叠加的关联记忆实验。
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
keys=np.array([[[[1.,0.],[1.,0.],[0.,1.]]]])
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
values=np.array([[[[1.,2.],[7.,8.],[3.,4.]]]])
# NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
_,memory=delta_recurrent(keys,keys,values,np.ones((1,1,3)))
# NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
check('associative overwrite two keys',memory[0,0],np.array([[7.,8.],[3.,4.]]))
print('All required CPU checks passed:',len(TESTS))

OPTIONAL=[]
if importlib.util.find_spec('torch') is None:
    OPTIONAL.append(('PyTorch SDPA','SKIP: torch not installed'))
    OPTIONAL.append(('official FA2 / FA3','SKIP: torch/CUDA unavailable'))
else:
    import torch
    import torch.nn.functional as F
    tq,tk,tv=[torch.tensor(a,dtype=torch.float64) for a in (q,k,v)]
    ty=F.scaled_dot_product_attention(tq,tk,tv,dropout_p=0.,is_causal=True)
    # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
    np.testing.assert_allclose(ty.numpy(),o,atol=1e-10,rtol=1e-10)
    OPTIONAL.append(('PyTorch SDPA CPU','PASS'))
    if not torch.cuda.is_available():
        OPTIONAL.append(('official FA2 / FA3','SKIP: CUDA unavailable'))
    else:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        torch.manual_seed(7)
        a,b,c=[torch.randn(1,128,2,64,device='cuda',dtype=torch.float16) for _ in range(3)]
        with sdpa_kernel(SDPBackend.MATH):
            # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
            ref=F.scaled_dot_product_attention(a.transpose(1,2).float(),b.transpose(1,2).float(),
                                              # NumPy：transpose 中每个整数指向原轴编号；按给出的顺序重新排列，数值本身不变。
                                              c.transpose(1,2).float(),is_causal=True).transpose(1,2)
        def verify_gpu(name,fn):
            out=fn(a,b,c,causal=True)
            if isinstance(out,tuple): out=out[0]
            torch.testing.assert_close(out.float(),ref,atol=4e-3,rtol=4e-3)
            for _ in range(3): fn(a,b,c,causal=True)
            torch.cuda.synchronize(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats(); start.record()
            for _ in range(10): fn(a,b,c,causal=True)
            end.record(); torch.cuda.synchronize()
            OPTIONAL.append((name,f'PASS: {start.elapsed_time(end)/10:.4f} ms; peak allocated {torch.cuda.max_memory_allocated()} bytes'))
        print('GPU:',torch.cuda.get_device_name(),'torch:',torch.__version__,'CUDA:',torch.version.cuda)
        if importlib.util.find_spec('flash_attn') is None:
            OPTIONAL.append(('official FA2','SKIP: flash_attn not installed'))
        else:
            from flash_attn import flash_attn_func
            verify_gpu('official flash_attn (inspect installed version)',flash_attn_func)
        if torch.cuda.get_device_capability()[0]!=9:
            OPTIONAL.append(('official FA3 Hopper','SKIP: this exercise targets Hopper SM90'))
        elif importlib.util.find_spec('flash_attn_interface') is None:
            OPTIONAL.append(('official FA3 Hopper','SKIP: hopper interface not installed'))
        else:
            from flash_attn_interface import flash_attn_func as fa3_func
            verify_gpu('official FA3 Hopper',fa3_func)
for name,status in OPTIONAL: print(name,':',status)

print('='*58)
print('REQUIRED CPU CHECKS:',len(TESTS),'PASS; failures: 0')
print('Code covers: Softmax, Linear, DeltaNet, GDN, KDA,')
print('FA1/2/3 educational models, PagedAttention, RadixAttention')
print('Optional checks:',OPTIONAL)
print('Run the complete reference script to reproduce.')
