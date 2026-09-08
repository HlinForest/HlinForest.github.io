"""CPU teaching examples: safe/online softmax, GQA, absorbed MLA and gathered DSA.

Run: python attention-variants.py. Only NumPy is required.
This tests algebra, not GPU speed or a pretrained DeepSeek checkpoint.
"""
import numpy as np


def safe_softmax(x):
    """Normalize the last axis; a completely masked row returns zeros."""
    # axis=-1 selects keys. keepdims retains a size-1 axis for row broadcasting.
    # NumPy：max 沿 axis 指定的轴取最大值；axis=-1 是最后一轴，keepdims=True 保留长度 1 的轴供广播。
    maximum = np.max(x, axis=-1, keepdims=True)
    # np.where(condition, yes, no) selects elementwise, without Python branching.
    # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
    # NumPy：isfinite 返回布尔数组，有限数为 True，正负无穷和 NaN 为 False。
    baseline = np.where(np.isfinite(maximum), maximum, 0.0)
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    weights = np.exp(x - baseline)  # exp(-inf)=0 for masked entries.
    # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
    denominator = weights.sum(axis=-1, keepdims=True)
    # where=False leaves the explicitly initialized output at zero: no 0/0.
    # NumPy：divide 在 where=True 处除法；False 处保留 out 的原值，所以这里用 zeros_like 初始化输出。
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    return np.divide(weights, denominator, out=np.zeros_like(weights),
                     where=denominator > 0)


def softmax_three_pass(x):
    """One finite, nonempty 1D score vector; no saved exponential array."""
    maximum = -np.inf
    for score in x:  # Pass 1: maximum.
        maximum = max(maximum, float(score))
    denominator = 0.0
    for score in x:  # Pass 2: denominator with the final common baseline.
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        denominator += np.exp(score - maximum)
    # NumPy：empty_like 只分配同形空间、不初始化数值；读取前必须写好每个元素。
    result = np.empty_like(x, dtype=float)  # Every entry is written below.
    for j, score in enumerate(x):  # Pass 3: output probabilities.
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        result[j] = np.exp(score - maximum) / denominator
    return result


def softmax_two_pass(x):
    """有限、非空的一维输入：先合并最大值/分母，再输出概率。mask 版本见 fused_value_scan。"""
    maximum, denominator = -np.inf, 0.0
    for score in x:
        next_maximum = max(maximum, float(score))
        # Old exponentials used the old maximum: rescale their accumulated sum.
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        correction = np.exp(maximum - next_maximum)
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        denominator = correction * denominator + np.exp(score - next_maximum)
        maximum = next_maximum
    # A second read is still needed when we explicitly require all probabilities.
    # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
    return np.exp(x - maximum) / denominator


def fused_value_scan(scores, values):
    """Consume score/value pairs once, including masked scores and empty tiles."""
    maximum, denominator = -np.inf, 0.0
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    numerator = np.zeros(values.shape[-1])  # One accumulator per value channel.
    for score, value in zip(scores, values):
        # NumPy：isneginf 判断是否为负无穷；这种分数代表被 mask 的位置。
        if np.isneginf(score):  # No contribution; avoids -inf - (-inf).
            continue
        next_maximum = max(maximum, float(score))
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        correction = np.exp(maximum - next_maximum)
        # NumPy：exp 对每个元素计算自然指数 e**x；输入输出形状相同，exp(-inf)=0。
        weight = np.exp(score - next_maximum)
        numerator = correction * numerator + weight * value
        denominator = correction * denominator + weight
        maximum = next_maximum
    return numerator / denominator if denominator > 0 else numerator


def gqa_without_repeat(q, k, v):
    """q [Hq,N,dk], k [Hkv,M,dk], v [Hkv,M,dv]; causal self-attention."""
    hq, n, dk = q.shape
    hkv, m, _ = k.shape
    assert n == m and hq % hkv == 0
    # arange generates positions [0,...,n-1]; None adds a row/column singleton.
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    keep = np.arange(m)[None, :] <= np.arange(n)[:, None]
    outputs = []
    for head in range(hq):
        group = head // (hq // hkv)  # Several query heads address one KV head.
        # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
        scores = q[head] @ k[group].T / np.sqrt(dk)
        # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
        probabilities = safe_softmax(np.where(keep, scores, -np.inf))
        outputs.append(probabilities @ v[group])
    # stack inserts a new head axis, unlike concatenate along an existing axis.
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    return np.stack(outputs, axis=0)


def rope_rows(x, positions):
    """Rotate adjacent coordinate pairs of x [tokens, even_dim]."""
    assert x.shape[-1] % 2 == 0
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    frequencies = 10000.0 ** (-np.arange(0, x.shape[-1], 2) / x.shape[-1])
    # Outer product [tokens,1]*[1,pairs] gives one angle per token/pair.
    angles = positions[:, None] * frequencies[None, :]
    # NumPy：cos 逐元素计算余弦，输入角度单位为弧度。
    # NumPy：sin 逐元素计算正弦，与余弦一起构造二维旋转。
    cosine, sine = np.cos(angles), np.sin(angles)
    # NumPy：empty_like 只分配同形空间、不初始化数值；读取前必须写好每个元素。
    result = np.empty_like(x)
    even, odd = x[:, 0::2], x[:, 1::2]  # Slice every second coordinate.
    result[:, 0::2] = even * cosine - odd * sine
    result[:, 1::2] = even * sine + odd * cosine
    return result


def mla(qc, qr, c, kr, uk, uv, wo, absorbed=True, selections=None):
    """One batch, causal MLA; C and the queries are already projected/normalized.

    qc [H,N,dc], qr [H,N,dR], c [M,r], kr [M,dR],
    uk [H,r,dc], uv [H,r,dv], wo [H*dv,D]. qr/kr already include RoPE.
    selections optionally lists source positions for each query (shared by heads).
    """
    heads, n, dc = qc.shape
    dv, model_width = uv.shape[-1], wo.shape[-1]
    assert n == c.shape[0] and qr.shape[-1] == kr.shape[-1]
    # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
    scale = np.sqrt(dc + qr.shape[-1])  # Preserve ORIGINAL concatenated width.
    # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
    output = np.zeros((n, model_width))
    probabilities = []
    for t in range(n):
        # With sparse selections, gather happens BEFORE the main dot products.
        # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
        indices = np.arange(t + 1) if selections is None else selections[t]
        # NumPy：unique 去重并排序；这里只用来断言所选历史位置没有重复。
        assert len(indices) > 0 and len(np.unique(indices)) == len(indices)
        # NumPy：all 检查布尔数组是否全部为真，返回真/假而非数值误差大小。
        assert np.all((indices >= 0) & (indices <= t))  # No future leakage.
        cs, rs = c[indices], kr[indices]  # Advanced integer indexing gathers rows.
        per_head = []
        for head in range(heads):
            if absorbed:
                q_latent = qc[head, t] @ uk[head].T  # [dc]@[dc,r] -> [r].
                content = q_latent @ cs.T  # [r]@[r,k] -> one score per selected key.
            else:
                expanded_keys = cs @ uk[head]  # [k,r]@[r,dc] -> [k,dc].
                content = qc[head, t] @ expanded_keys.T
            position = qr[head, t] @ rs.T
            prob = safe_softmax((content + position) / scale)
            if absorbed:
                latent_sum = prob @ cs  # Aggregate k histories to ONE [r] vector.
                head_output = latent_sum @ uv[head]  # Expand only the aggregate.
            else:
                expanded_values = cs @ uv[head]
                head_output = prob @ expanded_values
            # This row block is the portion of WO associated with this head.
            wo_head = wo[head * dv:(head + 1) * dv]
            output[t] += head_output @ wo_head
            per_head.append(prob)
        # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
        probabilities.append(np.stack(per_head))
    return output, probabilities


def dsa_indices(q_index, k_index, head_weights, top_k):
    """FP64 semantic indexer: qI [HI,N,dI], kI [N,dI], w [N,HI].

    This does not implement the official FP8 kernel or learned projections.
    Returned integer positions are shared across all MAIN MLA heads.
    """
    assert top_k > 0
    n = k_index.shape[0]
    selected = []
    for t in range(n):
        # [HI,dI]@[dI,t+1] -> one score row for each indexer head.
        dots = q_index[:, t, :] @ k_index[:t + 1].T
        # NumPy：maximum 比较对应位置取较大值，与沿整条轴归约的 max 不同。
        positive_dots = np.maximum(dots, 0.0)  # ReLU; weights themselves may be negative.
        # [HI,1] broadcasts over candidate positions; sum axis=0 removes HI.
        # NumPy：数组的 sum(axis, keepdims=...) 沿指定轴求和；-1 是末轴，-2 是倒数第二轴。
        scores = (head_weights[t, :, None] * positive_dots).sum(axis=0)
        count = min(top_k, t + 1)
        # argsort returns integer LOCATIONS, not sorted score values.
        # Stable full sort is an easy CPU oracle; production uses efficient top-k.
        # NumPy：argsort 返回排序后的整数位置；不返回分数本身，负号可将升序变为降序。
        by_score = np.argsort(-scores, kind='stable')[:count]
        # NumPy：sort 返回按数值升序排列的新数组；对选中位置排序仅调整一致的历史读取顺序。
        selected.append(np.sort(by_score))  # Temporal order helps inspect the gather.
    return selected


def run_checks():
    rng = np.random.default_rng(20260908)  # Local generator; no global state changes.
    passed = []
    def close(name, actual, expected):
        # NumPy：assert_allclose 比较每格误差是否 ≤ atol+rtol*abs(expected)；越界会立即报错。
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        passed.append(name)

    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    for scores in [np.array([0., 1., 2.]), np.array([1000., -1000., 999.])]:
        close('three-pass stable', softmax_three_pass(scores), safe_softmax(scores))
        close('two-pass stable', softmax_two_pass(scores), safe_softmax(scores))
        # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
        vals = rng.normal(size=(3, 4))
        close('fused value scan', fused_value_scan(scores, vals), safe_softmax(scores) @ vals)
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    # NumPy：full(shape,value) 把每一格初始化为同一个值，如行最大值从 -inf 开始。
    for scores in [np.array([-np.inf, 0., 2.]), np.full(3, -np.inf)]:
        # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
        vals = rng.normal(size=(3, 4))
        close('masked first/all scores', fused_value_scan(scores, vals), safe_softmax(scores) @ vals)
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('weighted sum hand example', np.array([.25, .75]) @ np.array([[1., 2.], [5., 6.]]), [4., 5.])
    # NumPy：outer(a,b) 生成 a_i*b_j 的二维表，两个向量轴都保留，是外积而非点积。
    memory = np.outer([1., 2.], [3., 1., 2.]) + np.outer([2., 1.], [1., 4., 2.])
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('linear state hand example', np.array([1., 1.]) @ memory / 6., [2., 2.5, 2.])

    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    gq = rng.normal(size=(4, 5, 3)); gk = rng.normal(size=(2, 5, 3)); gv = rng.normal(size=(2, 5, 2))
    shared = gqa_without_repeat(gq, gk, gv)
    # NumPy：repeat 沿指定轴重复元素/切片，确实会复制；这里只用于验证共享 KV 的基准。
    copied = gqa_without_repeat(gq, np.repeat(gk, 2, axis=0), np.repeat(gv, 2, axis=0))
    close('GQA shared addressing vs repeated oracle', shared, copied)

    # Deliberately unequal dc=4, r=3, dv=2: catches hidden square-shape mistakes.
    heads, n, dc, r, dv, dr, dim = 2, 7, 4, 3, 2, 2, 5
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    qc = rng.normal(size=(heads, n, dc)); c = rng.normal(size=(n, r))
    # NumPy：stack 在指定位置新建一条轴；把逐 token 的 [B,H,dv] 结果沿 -2 堆成 [B,H,N,dv]。
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    qr = np.stack([rope_rows(rng.normal(size=(n, dr)), np.arange(n)) for _ in range(heads)])
    # NumPy：arange(n) 生成整数位置 0…n-1；它表示索引，不是实际特征值。
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    kr = rope_rows(rng.normal(size=(n, dr)), np.arange(n))
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    uk = rng.normal(size=(heads, r, dc)); uv = rng.normal(size=(heads, r, dv))
    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    wo = rng.normal(size=(heads * dv, dim))
    args = (qc, qr, c, kr, uk, uv, wo)
    dense, _ = mla(*args, absorbed=False)
    close('MLA expanded vs absorbed with decoupled RoPE', mla(*args)[0], dense)
    close('value and WO weight absorption', (c @ uv[0]) @ wo[:dv], c @ (uv[0] @ wo[:dv]))

    # NumPy：normal(size=shape) 生成给定形状的正态样本；这里只用作可复现数学测试输入。
    iq = rng.normal(size=(3, n, 4)); ik = rng.normal(size=(n, 4)); iw = rng.normal(size=(n, 3))
    selections = dsa_indices(iq, ik, iw, top_k=3)
    sparse, _ = mla(*args, selections=selections)
    # Independent full-score-then-mask oracle; intentionally does NOT save work.
    # NumPy：zeros_like 生成与参照数组同形、同 dtype 的全零数组，用于输出或状态初始化。
    reference = np.zeros_like(sparse)
    for t in range(n):
        # NumPy：zeros(shape) 按给定形状申请全零数组；shape 元组中的顺序就是实际轴顺序。
        mask = np.zeros(n, dtype=bool); mask[selections[t]] = True
        for head in range(heads):
            # NumPy：sqrt 逐元素开平方；用于 dk 缩放或均方根归一化。
            logits = (qc[head, t] @ (c @ uk[head]).T + qr[head, t] @ kr.T) / np.sqrt(dc + dr)
            # NumPy：where(条件,真分支,假分支) 逐元素选择；两分支表达式都会先求值，不能靠它阻止不安全的 exp。
            prob = safe_softmax(np.where(mask, logits, -np.inf))
            reference[t] += (prob @ (c @ uv[head])) @ wo[head * dv:(head + 1) * dv]
    close('DSA gathered computation vs full masked oracle', sparse, reference)
    all_keys = dsa_indices(iq, ik, iw, top_k=n)
    close('DSA selecting full prefix reduces to dense MLA', mla(*args, selections=all_keys)[0], dense)
    # Perturb future cache and indexer keys; earlier selections/outputs must match.
    future_c, future_ik, future_kr = c.copy(), ik.copy(), kr.copy()
    future_c[4:] += 100.; future_ik[4:] -= 100.; future_kr[4:] += 30.
    future_selections = dsa_indices(iq, future_ik, iw, top_k=3)
    changed = mla(qc, qr, future_c, future_kr, uk, uv, wo, selections=future_selections)[0]
    close('DSA no future leakage', changed[:4], sparse[:4])
    # A negative head weight makes even a positive ReLU dot contribute negatively.
    # NumPy：ones(shape) 生成全 1 数组；用在 mask、乘法恒等门或数学参考值中。
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    signed = dsa_indices(np.ones((1, 2, 2)), np.array([[2., 2.], [1., 1.]]), -np.ones((2, 1)), 1)
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('signed indexer head weight', signed[1], np.array([1]))
    # Known top-k history slots used in the explanatory diagram.
    # NumPy：sort 返回按数值升序排列的新数组；对选中位置排序仅调整一致的历史读取顺序。
    # NumPy：argsort 返回排序后的整数位置；不返回分数本身，负号可将升序变为降序。
    # NumPy：array 将嵌套列表转为 ndarray；每层列表定义一条轴，含小数通常得到浮点 dtype。
    close('diagram top-k indices', np.sort(np.argsort(-np.array([8, 2, 9, 1, 7]))[:3]), [0, 2, 4])
    print(f'VARIANT CHECKS: {len(passed)} PASS; failures: 0')
    return passed


if __name__ == '__main__':
    run_checks()
