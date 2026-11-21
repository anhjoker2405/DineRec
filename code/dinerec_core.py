"""DineRec - lõi thuật toán (Thuật toán 2.2, 2.3, 2.4, 2.5 của báo cáo).
Phụ thuộc: numpy, scipy. Không phụ thuộc Surprise (Surprise chỉ dùng để đối chiếu trong tn8_real.py)."""
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------- Thuật toán 2.2
def pearson_blocks(R, min_support=2, block=1000):
    """Sinh (u0, Sim_block) với Sim_block là ma trận thưa CSR (hàng = người dùng trong khối, cột = mọi người dùng)
    chỉ chứa độ tương đồng Pearson DƯƠNG trên tập đồng đánh giá (đường chéo = 0). R: CSR |U|x|I|, ô thiếu = 0."""
    R = sp.csr_matrix(R, dtype=np.float64); R.sort_indices()
    Y = R.copy(); Y.data[:] = 1.0
    R2 = R.copy(); R2.data = R2.data ** 2
    YT, RT, R2T = Y.T.tocsc(), R.T.tocsc(), R2.T.tocsc()
    nU = R.shape[0]
    for u0 in range(0, nU, block):
        u1 = min(nU, u0 + block)
        N = (Y[u0:u1] @ YT).tocsr()
        S = (R[u0:u1] @ YT).tocsr()
        S2 = (Y[u0:u1] @ RT).tocsr()
        W = (R2[u0:u1] @ YT).tocsr()
        W2 = (Y[u0:u1] @ R2T).tocsr()
        P = (R[u0:u1] @ RT).tocsr()
        for M in (N, S, S2, W, W2, P): M.sort_indices()
        for M in (S, S2, W, W2, P):
            assert np.array_equal(M.indptr, N.indptr) and np.array_equal(M.indices, N.indices)
        n, s, s2, w, w2, p = N.data, S.data, S2.data, W.data, W2.data, P.data
        num = p - s * s2 / n
        den = np.sqrt(np.maximum(w - s * s / n, 0.0) * np.maximum(w2 - s2 * s2 / n, 0.0))
        sim = np.zeros_like(n)
        ok = (n >= min_support) & (den > 0)
        sim[ok] = num[ok] / den[ok]
        rows = np.repeat(np.arange(u0, u1), np.diff(N.indptr))
        sim[rows == N.indices] = 0.0            # đường chéo
        sim[sim <= 0] = 0.0                     # KNNWithMeans chỉ dùng láng giềng sim > 0
        out = sp.csr_matrix((sim, N.indices, N.indptr), shape=N.shape)
        out.eliminate_zeros()
        yield u0, out

# ---------------------------------------------------------------- Thuật toán 2.3 (một người dùng)
def knn_scores_user(u, sim_row, R_csc, user_mean, knn=20, min_k=1):
    """Trả về (items, rhat, has) cho mọi nhà hàng có ít nhất một láng giềng dương đã đánh giá, theo công thức KNNWithMeans.
    sim_row: hàng CSR (1 x |U|) chỉ chứa sim > 0. Nhà hàng không xuất hiện trong kết quả có has = False và rhat = r_bar_u."""
    nb = sim_row.indices; sv = sim_row.data
    if nb.size == 0:
        return np.array([], int), np.array([]), np.array([], bool)
    sub = R_csc  # |U| x |I| dạng CSC: lấy theo hàng nên dùng CSR của neighbors
    Rn = sub[nb].tocoo()                                 # hàng = láng giềng
    simv = sv[Rn.row]; item = Rn.col; dev = Rn.data - user_mean[nb][Rn.row]
    order = np.lexsort((nb[Rn.row], -simv, item))        # theo nhà hàng, sim giảm dần, rồi chỉ số người dùng tăng dần
    item, simv, dev = item[order], simv[order], dev[order]
    first = np.r_[0, np.flatnonzero(np.diff(item)) + 1]
    grp = np.repeat(np.arange(first.size), np.diff(np.r_[first, item.size]))
    rank = np.arange(item.size) - first[grp]
    keep = rank < knn
    item, simv, dev, grp = item[keep], simv[keep], dev[keep], grp[keep]
    ssum = np.bincount(grp, weights=simv, minlength=first.size)
    dsum = np.bincount(grp, weights=simv * dev, minlength=first.size)
    cnt = np.bincount(grp, minlength=first.size)
    items = item[np.r_[0, np.flatnonzero(np.diff(grp)) + 1]]
    rhat = np.full(first.size, user_mean[u])
    ok = cnt >= min_k
    rhat[ok] = user_mean[u] + dsum[ok] / ssum[ok]
    return items, rhat, cnt >= 1

def topk_two_tier(items, rhat, has, rated_mask, user_mean_u, K, item_order):
    """Khóa (2.8): khóa = (có láng giềng, r_hat chưa cắt, -thứ tự business_id). rated_mask: các nhà hàng u đã đánh giá."""
    nI = rated_mask.size
    r = np.full(nI, user_mean_u); h = np.zeros(nI, bool)
    r[items] = rhat; h[items] = has
    cand = np.flatnonzero(~rated_mask)
    key = np.lexsort((item_order[cand], -r[cand], -h[cand].astype(int)))
    return cand[key][:K], r, h

# ---------------------------------------------------------------- Thuật toán 2.4
def weiszfeld(X, eps=1e-3, tmax=100):
    """Trung vị hình học (Vardi-Zhang) trên mặt phẳng (km). X: (n,2)."""
    n = len(X)
    if n <= 2: return X.mean(0)
    z = X.mean(0)
    for _ in range(tmax):
        dist = np.linalg.norm(X - z, axis=1)
        D = dist < 1e-9; eta = int(D.sum())
        nd = ~D
        w = 1.0 / dist[nd]
        T = (X[nd] * w[:, None]).sum(0) / w.sum() if nd.any() else z
        if eta == 0:
            zn = T
        else:
            gam = np.linalg.norm(((X[nd] - z) / dist[nd][:, None]).sum(0))
            if gam <= eta: break
            zn = (1 - eta / gam) * T + (eta / gam) * z
        if np.linalg.norm(zn - z) < eps: z = zn; break
        z = zn
    return z

def project(lat, lon, lat0, lon0, rho=6371.0):
    p, t, p0, t0 = np.radians(lat), np.radians(lon), np.radians(lat0), np.radians(lon0)
    return np.c_[rho * (t - t0) * np.cos(p0), rho * (p - p0)]

def haversine(lat1, lon1, lat2, lon2, rho=6371.0):
    p1, p2, t1, t2 = map(np.radians, (lat1, lat2, lon1, lon2))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((t2 - t1) / 2) ** 2
    return 2 * rho * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))   # kẹp về [0,1] trước khi lấy arcsin

# ---------------------------------------------------------------- Thuật toán 2.5
def geo_rerank(rhat, d, K, lam, kind='minmax', sigma=None, ids=None, has=None):
    """rhat, d: điểm chưa cắt và khoảng cách (km) của M ứng viên; trả về chỉ số K phần tử có điểm kết hợp lớn nhất.
    has: cờ 'có láng giềng'. Ứng viên không có bằng chứng CF nhận s = 0; min-max của s chỉ tính trên ứng viên có bằng chứng;
    khi hòa điểm: ứng viên có bằng chứng trước, rồi r_hat giảm dần, rồi business_id. Nhờ vậy lambda = 1 trùng đúng thứ tự của mô hình nền (khóa 2.8)."""
    if has is None: has = np.ones(len(rhat), bool)
    rhat = np.round(rhat, 9)                        # cùng cách làm tròn với rank_all: bỏ nhiễu dấu phẩy động (hòa thật xếp theo business_id)
    if has.any():
        cmin, cmax = rhat[has].min(), rhat[has].max()
        s = (rhat - cmin) / (cmax - cmin) if cmax > cmin else np.ones_like(rhat)
        s = np.where(has, s, 0.0)
    else:
        s = np.zeros_like(rhat)
    if kind == 'minmax':
        dmin, dmax = d.min(), d.max()
        g = 1 - (d - dmin) / (dmax - dmin) if dmax > dmin else np.ones_like(d)
    else:
        g = np.exp(-d / max(sigma, 1.0))
    score = lam * s + (1 - lam) * g
    tie = np.arange(len(score)) if ids is None else ids
    return np.lexsort((tie, -rhat, -has.astype(int), -score))[:min(K, len(score))], s, g
