"""TN8 trên dữ liệu thật: Surprise KNNWithMeans (user-based, Pearson) so với Thuật toán 2.2-2.3.
Dùng: python tn8_real.py REVIEWS BUSINESS MODE N [TAU K]
  MODE = surprise | ours | check   ;  N = số người dùng lấy mẫu từ H_val (0 = toàn bộ)
Mỗi lần gọi là một tiến trình riêng để đo bộ nhớ đỉnh (ru_maxrss). In một dòng JSON."""
import sys, time, json, resource, argparse
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, '.')
from dinerec_core import pearson_blocks, knn_scores_user

rev, biz, mode, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tau = int(sys.argv[5]) if len(sys.argv) > 5 else 5
knn = int(sys.argv[6]) if len(sys.argv) > 6 else 5
rv = pd.read_csv(rev, parse_dates=['date']); rv = rv[rv.date >= '2018-01-01'].sort_values(['date', 'review_id'])
ds = rv.date.sort_values().reset_index(drop=True); T1 = ds.iloc[int(len(ds) * .8) - 1]
H = rv[rv.date <= T1]
cnt = H.groupby('business_id').user_id.nunique(); H = H[H.business_id.isin(cnt[cnt >= 5].index)]
users = np.array(sorted(H.user_id.unique()))
if N > 0:
    users = np.random.default_rng(0).choice(users, size=N, replace=False)
    H = H[H.user_id.isin(set(users))]
from surprise import Dataset, Reader, KNNWithMeans
ts = Dataset.load_from_df(H[['user_id', 'business_id', 'stars']], Reader(rating_scale=(1, 5))).build_full_trainset()
a = np.array([(u, i, r) for u, i, r in ts.all_ratings()])
R = sp.csr_matrix((a[:, 2], (a[:, 0].astype(int), a[:, 1].astype(int))), shape=(ts.n_users, ts.n_items)); R.sort_indices()
nU, nI = ts.n_users, ts.n_items
colcnt = np.diff(R.tocsc().indptr).astype(float)
info = dict(mode=mode, n_users=nU, n_items=nI, ratings=int(R.nnz), sum_Ui2=float((colcnt ** 2).sum()), nU2=float(nU) ** 2, tau=tau, k=knn)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
t = time.time()
if mode == 'surprise':
    KNNWithMeans(k=knn, min_k=1, sim_options={'name': 'pearson', 'user_based': True, 'min_support': tau}, verbose=False).fit(ts)
elif mode == 'ours':
    nnz = 0
    for u0, S in pearson_blocks(R, tau, block=1000): nnz += S.nnz
    info['sim_nnz_positive'] = nnz
elif mode == 'check':
    al = KNNWithMeans(k=knn, min_k=1, sim_options={'name': 'pearson', 'user_based': True, 'min_support': tau}, verbose=False); al.fit(ts)
    umean = np.asarray(R.sum(1)).ravel() / np.maximum(1, R.getnnz(axis=1))
    maxdsim = 0.0; rows = {}
    for u0, S in pearson_blocks(R, tau, block=1000):
        D = S.toarray(); ref = np.where(al.sim[u0:u0 + D.shape[0]] > 0, al.sim[u0:u0 + D.shape[0]], 0.0)
        ref[np.arange(D.shape[0]), np.arange(u0, u0 + D.shape[0])] = 0
        maxdsim = max(maxdsim, float(np.abs(D - ref).max()))
        for k in range(D.shape[0]): rows[u0 + k] = S[k]
    rng = np.random.default_rng(1); maxdr = 0.0; nmis = 0; ntot = 0
    for u in rng.choice(nU, size=min(50, nU), replace=False):
        items, rhat, has = knn_scores_user(int(u), rows[int(u)], R, umean, knn, 1)
        got = dict(zip(items.tolist(), rhat)); rated = set(R.indices[R.indptr[u]:R.indptr[u + 1]].tolist())
        for i in range(nI):
            if i in rated: continue
            est = al.estimate(int(u), i); est = est[0] if isinstance(est, tuple) else est
            mine = got.get(i, umean[u]); ntot += 1
            if abs(est - mine) > 1e-9: nmis += 1; maxdr = max(maxdr, abs(est - mine))
    info.update(max_dsim=maxdsim, max_drhat=maxdr, n_mismatch=nmis, n_pairs=ntot)
info['sec'] = round(time.time() - t, 2)
info['peak_GB'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
info['peak_before_GB'] = round(base, 3)
print(json.dumps(info), flush=True)
