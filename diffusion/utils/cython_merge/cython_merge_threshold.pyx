# merge_threshold.pyx
import numpy as np
cimport numpy as np
cimport cython

cpdef merge_cython_threshold(double[:,:] adj_mat, double alpha, long seed):
    """
    阶段1：仅贪心插入 adj_mat[i,j] > alpha 的边（从大到小），避免冲突与早期成环；
    阶段2：将阶段1得到的多个子路径随机拼接成一条完整路径，并首尾闭合为回路。
    返回：A(0/1 邻接矩阵), merge_iterations(枚举的边数)
    """
    cdef long N = adj_mat.shape[0]
    cdef double[:,:] A = np.zeros((N, N), dtype=np.float64)

    # 记录每个点所在“链”的头、尾（并查集风格）
    cdef int[:] route_begin = np.arange(N, dtype='int32')
    cdef int[:] route_end   = np.arange(N, dtype='int32')

    cdef np.ndarray flat = np.asarray(adj_mat).flatten()
    cdef np.ndarray[np.int32_t, ndim=1] sorted_edges = np.argsort(-flat).astype('int32')

    cdef long merge_iterations = 0
    cdef long merge_count = 0
    cdef long total_needed = N - 1  # 成单条路径需要 N-1 条边

    cdef int i, j, edge
    cdef double w

    # ---------- 阶段1：阈值内贪心插入 ----------
    for edge in sorted_edges:
        merge_iterations += 1
        i = int(edge // N)
        j = int(edge % N)
        w = flat[edge]
        if w <= alpha:
            # 剩余边都不大于 alpha，直接退出
            break

        # 找各自当前链的首尾
        begin_i = find_begin(route_begin, i)
        end_i   = find_end(route_end, i)
        begin_j = find_begin(route_begin, j)
        end_j   = find_end(route_end, j)

        # 同一条链就跳过（避免短环）
        if begin_i == begin_j:
            continue

        # i、j 必须在各自链的首或尾，避免度数超过 2
        if i != begin_i and i != end_i:
            continue
        if j != begin_j and j != end_j:
            continue

        # 插边
        A[i, j] = 1.0
        A[j, i] = 1.0
        merge_count += 1

        # 按 4 种相对位置合并两条链（保持 begin/end 的不变式）
        if i == begin_i and j == end_j:
            # (head_i) --- ...  +  ... --- (tail_j)
            route_begin[begin_i] = begin_j
            route_end[end_j] = end_i

        elif i == end_i and j == begin_j:
            # (tail_i) --- ...  +  ... --- (head_j)
            route_begin[begin_j] = begin_i
            route_end[end_i] = end_j

        elif i == begin_i and j == begin_j:
            # head_i + head_j
            route_begin[begin_i] = end_j
            route_begin[begin_j] = end_j
            route_begin[end_j] = end_j
            route_end[end_j] = end_i
            route_end[begin_j] = end_i

        elif i == end_i and j == end_j:
            # tail_i + tail_j
            route_end[end_i] = begin_j
            route_begin[begin_j] = begin_i
            route_begin[end_j] = begin_i
            route_end[end_j] = begin_j
            route_end[begin_j] = begin_j

        # 若已经形成单条路径（N-1 条边）可早停
        if merge_count >= total_needed:
            break

    # ---------- 阶段2：随机拼接所有子路径 ----------
    # 收集所有“链”的 (begin, end)；孤立点会是 (i, i)
    cdef dict seen = {}
    cdef int u
    for u in range(N):
        b = find_begin(route_begin, u)
        seen[b] = 1  # 以链的 begin 作为代表

    comp_begins = list(seen.keys())              # 各子路径的 begin
    # 求每条子路径对应的 end
    comp_pairs = []
    for b in comp_begins:
        comp_pairs.append((b, find_end(route_end, b)))

    # 若阶段1已经构成单条路径，仍需闭合成回路
    cdef Py_ssize_t K = len(comp_pairs)
    import numpy as _np
    rng = _np.random.default_rng(int(seed) if seed is not None else None)
    if K > 1:
        # 打乱子路径顺序
        order = rng.permutation(K)
        # 依次把上一条的 end 连接到下一条的 begin
        for idx in range(K - 1):
            b1, e1 = comp_pairs[int(order[idx])]
            b2, e2 = comp_pairs[int(order[idx + 1])]
            A[e1, b2] = 1.0
            A[b2, e1] = 1.0
            # 更新 route_begin/route_end 的等价类（可选，但不再依赖它们了）
    else:
        order = _np.array([0], dtype=int)

    # 现在所有点构成一条大路径，把最后一条的 end 接回第一条的 begin，闭合为回路
    # （若 K==1 也会把该路径的两端闭合）
    b_first, e_first = comp_pairs[int(order[0])]
    b_last,  e_last  = comp_pairs[int(order[-1])]
    A[e_last, b_first] = 1.0
    A[b_first, e_last] = 1.0

    return A, merge_iterations


# ====== 复用你现有的并查集辅助函数 ======
cpdef int find_begin(int[:] route_begin, int i):
    cdef int bi = route_begin[i]
    if bi != i:
        bi = find_begin(route_begin, bi)
        route_begin[i] = bi
    return bi

cpdef int find_end(int[:] route_end, int i):
    cdef int ei = route_end[i]
    if ei != i:
        ei = find_end(route_end, ei)
        route_end[i] = ei
    return ei