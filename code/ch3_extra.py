"""Các số bổ sung của Chương 3 (mục 3.3, 3.4, 3.5, 3.6, 3.9, 3.12). Không chọn lại gì: đọc quyết định từ decisions_from_val.json.
Dùng: python ch3_extra.py REVIEWS BUSINESS RESULTS_DIR OUT_DIR   -> OUT_DIR/extra.json"""
import sys, json, argparse, os
import numpy as np, pandas as pd
sys.path.insert(0, '.')
import run_experiments as X

rev, biz, res, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
dec = json.load(open(f'{res}/decisions_from_val.json'))
cfg = {k: tuple(v[0]) for k, v in dec['best'].items()}; lam = dec['lam']['main']; lamP = dec['lam_P']
E = {}

# ---- tỷ lệ người dùng có NDCG@10 dương (mục 3.4, 3.12)
p = pd.read_csv(f'{res}/TN7_per_user.csv')
E['share_nonzero_base'] = float((p.ndcg10_base > 0).mean()); E['share_nonzero_rr'] = float((p.ndcg10_rerank > 0).mean()); E['share_nonzero_pn'] = float((p.ndcg10_popnear > 0).mean())

# ---- so sánh thăm dò theo nhóm lịch sử (mục 3.9), không hiệu chỉnh Holm
coh = {}
for lab, (lo, hi) in {'1-4': (1, 4), '5-19': (5, 19), '>=20': (20, 10 ** 9)}.items():
    m = (p.n_hist >= lo) & (p.n_hist <= hi)
    for nm, (a, b) in {'rr_vs_l1': ('ndcg10_rerank', 'ndcg10_base'), 'dcf': ('ndcg10_rerank', 'ndcg10_popnear')}.items():
        r = X.compare(p[m][a].values, p[m][b].values, 1000)
        coh[f'{lab}:{nm}'] = {k: float(v) for k, v in r.items()}
E['cohort_tests'] = coh

a = argparse.Namespace(reviews=rev, business=biz, start='2018-01-01', nmin=5)
rv, bz, tr, va, te, T1, T2 = X.load_split(a)

# ---- giai đoạn chọn: hạt giống Biased MF trên Val, tỷ lệ người dùng có láng giềng UserKNN (mục 3.3)
sv = X.Stage(tr, va, bz, 5)
def val_ndcg(Rh, Has):
    rk, _ = X.rank_all(sv, Rh, Has); return float(X.per_user(sv, rk, 10)[2].mean())
E['bmf_seed_val'] = {str(sd): val_ndcg(*X.m_bmf(sv, *cfg['BiasedMF'], seed=sd)) for sd in X.SEEDS}
rows = X.user_sims(sv, cfg['UserKNN'][1]); E['users_with_nb_val'] = float(np.mean([rows[int(u)].nnz > 0 for u in sv.eu]))
# UserKNN theo k trên Val: danh sách top-10 và vị trí trúng có đổi không (mục 3.3)
ref = None; same_list = []; same_hit = True
for k in (5, 10, 20, 40, 80):
    Rh_, Has_ = X.m_userknn(sv, k, cfg['UserKNN'][1]); L_ = X.rank_all(sv, Rh_, Has_)[0][:, :10]; hit_ = np.take_along_axis(sv.Rel, L_, 1)
    if ref is None: ref = (L_.copy(), hit_.copy())
    same_list.append(float((L_ == ref[0]).all(1).mean())); same_hit = same_hit and bool((hit_ == ref[1]).all())
E['userknn_k_share_users_same_top10'] = min(same_list); E['userknn_k_same_hit_pattern'] = same_hit

# ---- giai đoạn báo cáo (mục 3.4, 3.6)
st = X.Stage(pd.concat([tr, va]), te, bz, 5)
rows = X.user_sims(st, cfg['UserKNN'][1]); E['users_with_nb_test'] = float(np.mean([rows[int(u)].nnz > 0 for u in st.eu]))
# độ phủ top-10 của UserKNN tách theo người dùng có / không có láng giềng trong danh sách (mục 3.4)
RhU, HasU = X.m_userknn(st, *cfg['UserKNN']); LU = X.rank_all(st, RhU, HasU)[0][:, :10]; fb = ~np.take_along_axis(HasU, LU, 1).any(1)
E['uknn_full_fallback_share'] = float(fb.mean()); E['uknn_cov_fallback_items'] = int(np.unique(LU[fb]).size); E['uknn_fallback_distinct_lists'] = len({tuple(x) for x in LU[fb]})
E['uknn_cov_nb_users'] = float(np.unique(LU[~fb]).size / st.R.shape[1]); E['uknn_cov_fallback'] = float(np.unique(LU[fb]).size / st.R.shape[1])
Rh, Has = X.m_itemknn(st, *cfg['ItemKNN']); rk, nc = X.rank_all(st, Rh, Has); Dm = st.dist('mean')
for key, l_ in (('l1', 1.0), ('rr', lam), ('l0', 0.0)):
    L = X.rerank(st, rk, nc, Rh, l_, Has=Has); d = np.take_along_axis(Dm, L[:, :10], 1)
    E[f'le5_{key}'] = float((d <= 5).mean()); E[f'p90_{key}'] = float(np.quantile(d, .9))
Rp = X.m_mostpop(st); rkp, ncp = X.rank_all(st, *Rp); Lp = X.rerank(st, rkp, ncp, Rp[0], lamP); d = np.take_along_axis(Dm, Lp[:, :10], 1)
E['le5_pn'] = float((d <= 5).mean())
ref = Dm[st.Rel]; E['le5_ref'] = float((ref <= 5).mean())
E['avgpop_rel'] = float(st.pop_all[np.nonzero(st.Rel)[1]].mean()); E['avgpop_cat'] = float(st.pop_all.mean())
# tập ứng viên theo độ phổ biến (PopNear) so với ItemKNN: người dùng không có nhà hàng liên quan trong M ứng viên (mục 3.6)
nrel = st.Rel.sum(1)
for nm, r_ in (('itemknn', rk), ('mostpop', rkp)):
    for M in (50, 100):
        h = np.array([st.Rel[e, r_[e, :M]].sum() for e in range(st.nE)])
        E[f'cand_{nm}_{M}_no_rel'] = float((h == 0).mean()); E[f'cand_{nm}_{M}_recall'] = float((h / nrel).mean())
E['min_candidates'] = int(nc.min()); E['short_lists'] = int((nc < 20).sum())
# biến thể mindist: khoảng cách theo định nghĩa riêng của nó (mục 3.8)
Dmin = st.dist('mindist'); Lmd = X.rerank(st, rk, nc, Rh, dec['lam']['mindist'], anchor='mindist', Has=Has); L1 = X.rerank(st, rk, nc, Rh, 1.0, Has=Has)
E['mindist_own_rr'] = float(np.take_along_axis(Dmin, Lmd[:, :10], 1).mean()); E['mindist_own_l1'] = float(np.take_along_axis(Dmin, L1[:, :10], 1).mean())
# hình học của ba người dùng tiêu biểu (mục 3.11), chọn theo đúng tiêu chí của ch3_analysis.py
from dinerec_core import haversine, project
nh = st.R.getnnz(axis=1)[st.eu]
def sep(e):
    u = st.eu[e]; J = st.R.indices[st.R.indptr[u]:st.R.indptr[u + 1]]
    if J.size < 6: return 0.0, None
    P = project(st.lat[J], st.lon[J], st.lat[J].mean(), st.lon[J].mean()); c0, c1 = P[np.argmin(P[:, 0])], P[np.argmax(P[:, 0])]
    for _ in range(20):
        lab = np.linalg.norm(P - c0, axis=1) > np.linalg.norm(P - c1, axis=1)
        if lab.all() or (~lab).all(): return 0.0, None
        c0, c1 = P[~lab].mean(0), P[lab].mean(0)
    return (float(np.linalg.norm(c0 - c1)), (c0, c1, int((~lab).sum()), int(lab.sum()))) if min(lab.sum(), (~lab).sum()) >= 3 else (0.0, None)
S = [sep(e) for e in range(st.nE)]; eA = int(np.argmax([s_[0] for s_ in S])); c0, c1, n0_, n1_ = S[eA][1]
E['caseA'] = dict(sep_km=S[eA][0], c0_north_km=float(c0[1]), c1_north_km=float(c1[1]), c0_east_km=float(c0[0]), c1_east_km=float(c1[0]), n0=n0_, n1=n1_)
ctr = (float(bz.latitude.median()), float(bz.longitude.median()))
cb = np.flatnonzero((nh <= 4) & (nrel >= 3)); eB = int(cb[np.lexsort((st.eu[cb], -nrel[cb]))][0])
cc = np.flatnonzero((nh >= 10) & (st.umean[st.eu] >= 4.8)); eC = int(cc[np.lexsort((st.eu[cc], -nh[cc]))][0])
for nm, e in (('caseB', eB), ('caseC', eC)):
    u = st.eu[e]; J = st.R.indices[st.R.indptr[u]:st.R.indptr[u + 1]]; la0, lo0 = st.lat[J].mean(), st.lon[J].mean(); Rl = np.flatnonzero(st.Rel[e])
    dh = haversine(la0, lo0, st.lat[J], st.lon[J])
    E[nm] = dict(n_hist=int(J.size), n_rel=int(Rl.size), anchor_to_centre_km=float(haversine(ctr[0], ctr[1], la0, lo0)), hist_median_km=float(np.median(dh)), hist_max_km=float(dh.max()),
                 rel_within3km=float((Dm[e, Rl] <= 3).mean()))
json.dump(E, open(f'{out}/extra.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in E.items() if k != 'cohort_tests'}, indent=1))

# ---- bước λ = 1 -> 0,9 của mô hình nền (mục 3.6): NDCG, số người dùng đổi, số nhà hàng bị thay, hạng gốc của lượt trúng mới
def step09(s_):
    Rh_, Has_ = X.m_itemknn(s_, *cfg['ItemKNN']); rk_, nc_ = X.rank_all(s_, Rh_, Has_)
    A, B = X.rerank(s_, rk_, nc_, Rh_, 1.0, Has=Has_), X.rerank(s_, rk_, nc_, Rh_, 0.9, Has=Has_)
    na, nb = X.per_user(s_, A, 10)[2], X.per_user(s_, B, 10)[2]; chg = []; gr = []
    for e in range(s_.nE):
        new = set(B[e, :10]) - set(A[e, :10]); chg.append(len(new)); pos = {j: i for i, j in enumerate(rk_[e, :min(100, nc_[e])])}
        gr += [pos[j] + 1 for j in new if s_.Rel[e, j]]
    return dict(ndcg1=float(na.mean()), ndcg09=float(nb.mean()), better=int((nb > na).sum()), worse=int((nb < na).sum()),
                changed=float(np.mean(chg)), newhit_rank_median=float(np.median(gr)), newhits=len(gr))
E2 = {'step09_val': step09(sv), 'step09_test': step09(st)}
E.update(E2); json.dump(E, open(f'{out}/extra.json', 'w'), indent=1); print(E2)

# ---- so với MostPopular theo nhóm lượng lịch sử (mục 3.5), thăm dò, không hiệu chỉnh Holm
def cohort_vs_mostpop(s_, p_):
    # làm tròn 12 chữ số: q đọc từ CSV, mp tính lại trong bộ nhớ; không làm tròn thì sai số ~1e-17 khiến Wilcoxon coi hiệu bằng 0 là khác 0
    Rp_ = X.m_mostpop(s_); rk_, nc_ = X.rank_all(s_, *Rp_)
    mp = X.per_user(s_, rk_[:, :10], 10)[2]
    uid = [s_.ts.to_raw_uid(int(u)) for u in s_.eu]; mp = pd.Series(mp, index=uid)
    q = p_.set_index('user'); assert (q.index == mp.index).all() if len(q) == len(mp) else False
    out_ = {}
    for lab, (lo, hi) in {'1-4': (1, 4), '5-19': (5, 19), '>=20': (20, 10 ** 9), 'all': (0, 10 ** 9)}.items():
        m = (q.n_hist >= lo) & (q.n_hist <= hi)
        for nm, col in (('rr_vs_mp', 'ndcg10_rerank'), ('base_vs_mp', 'ndcg10_base'), ('pn_vs_mp', 'ndcg10_popnear')):
            out_[f'{lab}:{nm}'] = {k: float(v) for k, v in X.compare(np.round(q[m][col].values, 12), np.round(mp[m.values].values, 12), 1000).items()}
    return out_
E.update({'cohort_vs_mostpop': cohort_vs_mostpop(st, p)}); json.dump(E, open(f'{out}/extra.json', 'w'), indent=1)
