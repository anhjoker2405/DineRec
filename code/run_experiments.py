"""DineRec - chạy thí nghiệm TN1-TN7 theo đúng quy trình Thuật toán 2.1 của báo cáo.

Dùng:  python run_experiments.py --reviews indy_reviews.csv --business indy_business.csv --out results
       python run_experiments.py ... --quick        (chạy thử nhanh: lưới nhỏ, B=200)

Cần: pip install scikit-surprise scipy pandas numpy   (đặt file này cạnh dinerec_core.py)
Mọi quyết định (cấu hình, mô hình nền, lambda) được chốt trên Val; Test chỉ dùng để báo cáo, không dùng để chọn (kết quả Test được tính ở cả ba vòng tinh chỉnh, xem Bảng 3.22).
"""
import argparse, os, time, itertools, json
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import norm, wilcoxon
from surprise import Dataset, Reader, SVD, BaselineOnly, KNNWithMeans
from dinerec_core import pearson_blocks, knn_scores_user, weiszfeld, project, haversine, geo_rerank

KS = (5, 10, 20)
LAMS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
SEEDS = (17, 42, 2026)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ============================================================ dữ liệu và chia tập (mục 2.3)
def load_split(a):
    rv = pd.read_csv(a.reviews, parse_dates=['date']); bz = pd.read_csv(a.business)
    rv = rv[rv.date >= a.start].sort_values(['date', 'review_id'])
    ds = rv.date.sort_values().reset_index(drop=True)
    T1, T2 = ds.iloc[int(len(ds) * .8) - 1], ds.iloc[int(len(ds) * .9) - 1]
    tr, va, te = rv[rv.date <= T1], rv[(rv.date > T1) & (rv.date <= T2)], rv[rv.date > T2]
    log(f'T1={T1.date()} T2={T2.date()} review={len(rv)} user={rv.user_id.nunique()} biz={rv.business_id.nunique()}')
    return rv, bz, tr, va, te, T1, T2


class Stage:
    """Một giai đoạn: lịch sử H, danh mục I_H, tập đo. Mọi chỉ số người dùng/nhà hàng là chỉ số trong của Surprise."""
    def __init__(s, H, E, bz, nmin):
        cnt = H.groupby('business_id').user_id.nunique(); keep = cnt[cnt >= nmin].index
        H = H[H.business_id.isin(keep)]
        s.ts = Dataset.load_from_df(H[['user_id', 'business_id', 'stars']], Reader(rating_scale=(1, 5))).build_full_trainset()
        ts = s.ts; s.nU, s.nI = ts.n_users, ts.n_items
        s.uid = {ts.to_raw_uid(i): i for i in range(s.nU)}; s.items = [ts.to_raw_iid(i) for i in range(s.nI)]
        iid = {r: i for i, r in enumerate(s.items)}
        a = np.array([(u, i, r) for u, i, r in ts.all_ratings()])
        s.R = sp.csr_matrix((a[:, 2], (a[:, 0].astype(int), a[:, 1].astype(int))), shape=(s.nU, s.nI)); s.R.sort_indices()
        s.umean = np.asarray(s.R.sum(1)).ravel() / np.maximum(1, s.R.getnnz(axis=1))
        s.pop_pos = np.asarray((s.R >= 4).sum(0)).ravel().astype(float)          # pop+ theo (2.4), mục 2.4.1
        s.pop_all = s.R.getnnz(axis=0).astype(float)
        b = bz.set_index('business_id').loc[s.items]; s.lat, s.lon = b.latitude.values, b.longitude.values
        s.order = np.argsort(np.argsort(np.array(s.items)))                      # thứ tự business_id (phá hòa)
        E = E[E.user_id.isin(s.uid) & E.business_id.isin(iid)].copy()
        E['u'] = E.user_id.map(s.uid); E['i'] = E.business_id.map(iid)
        seen = set(zip(*s.R.nonzero())); E = E[[(u, i) not in seen for u, i in zip(E.u, E.i)]]
        pos = E[E.stars >= 4]
        s.eu = np.array(sorted(pos.u.unique())); s.nE = len(s.eu); pos_e = {u: k for k, u in enumerate(s.eu)}
        s.Rel = np.zeros((s.nE, s.nI), bool)
        for u, i in zip(pos.u, pos.i): s.Rel[pos_e[u], i] = True
        s.Rseen = (s.R[s.eu] > 0).toarray()                                      # I_u^H của người dùng đo
        s.E_pairs = E[E.u.isin(pos_e)]                                           # để tính RMSE/MAE
        s.pos_e = pos_e
        g = s.R.getnnz(axis=1)[s.eu]; s.cohort = np.where(g <= 4, 0, np.where(g <= 19, 1, 2))
        s._D = {}
        log(f'  stage: train_user={s.nU} catalogue={s.nI} eval_users={s.nE} positives={int(s.Rel.sum())} cohorts={np.bincount(s.cohort, minlength=3)}')

    def dist(s, anchor):
        """Ma trận khoảng cách (nE x nI, km) từ vị trí đại diện đến mọi nhà hàng. anchor: mean | geomedian | mindist."""
        if anchor in s._D: return s._D[anchor]
        D = np.zeros((s.nE, s.nI))
        for e, u in enumerate(s.eu):
            J = s.R.indices[s.R.indptr[u]:s.R.indptr[u + 1]]
            la, lo = s.lat[J], s.lon[J]
            if anchor == 'mindist':
                D[e] = haversine(la[None, :], lo[None, :], s.lat[:, None], s.lon[:, None]).min(1)
                continue
            la0, lo0 = la.mean(), lo.mean()
            if anchor == 'geomedian':
                z = weiszfeld(project(la, lo, la0, lo0)); rho = 6371.0
                la0 = la0 + np.degrees(z[1] / rho); lo0 = lo0 + np.degrees(z[0] / (rho * np.cos(np.radians(la0))))
            D[e] = haversine(la0, lo0, s.lat, s.lon)
        s._D[anchor] = D; return D

    def sigma(s):
        """sigma_u = trung vị khoảng cách từ vị trí đại diện đến các nhà hàng đã đánh giá, sàn 1 km."""
        out = np.zeros(s.nE)
        for e, u in enumerate(s.eu):
            J = s.R.indices[s.R.indptr[u]:s.R.indptr[u + 1]]
            la0, lo0 = s.lat[J].mean(), s.lon[J].mean()
            out[e] = max(1.0, np.median(haversine(la0, lo0, s.lat[J], s.lon[J])))
        return out


# ============================================================ mô hình (mục 2.4): mỗi mô hình trả (Rhat, Has) cho người dùng đo
def m_random(s, seed=17):
    return np.random.default_rng(seed).random((s.nE, s.nI)), np.ones((s.nE, s.nI), bool)

def m_globalmean(s):
    return np.full((s.nE, s.nI), s.ts.global_mean), np.ones((s.nE, s.nI), bool)

def m_mostpop(s):
    return np.tile(s.pop_pos, (s.nE, 1)), np.ones((s.nE, s.nI), bool)

def m_baseline(s):
    al = BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 10, 'reg_u': 15, 'reg_i': 10}, verbose=False); al.fit(s.ts)
    return s.ts.global_mean + al.bu[s.eu][:, None] + al.bi[None, :], np.ones((s.nE, s.nI), bool)

def m_bmf(s, factors, reg, epochs, lr=0.005, seed=42):
    al = SVD(n_factors=factors, n_epochs=epochs, lr_all=lr, reg_all=reg, init_std_dev=0.1, random_state=seed, biased=True, verbose=False)
    al.fit(s.ts)
    return s.ts.global_mean + al.bu[s.eu][:, None] + al.bi[None, :] + al.pu[s.eu] @ al.qi.T, np.ones((s.nE, s.nI), bool)

def m_itemknn(s, k, tau, min_k=1):
    al = KNNWithMeans(k=k, min_k=min_k, sim_options={'name': 'pearson', 'user_based': False, 'min_support': tau}, verbose=False); al.fit(s.ts)
    sim, mean = al.sim, np.asarray(al.means)
    Rh = np.tile(mean, (s.nE, 1)); Has = np.zeros((s.nE, s.nI), bool)
    for e, u in enumerate(s.eu):
        J = s.R.indices[s.R.indptr[u]:s.R.indptr[u + 1]]; rj = s.R.data[s.R.indptr[u]:s.R.indptr[u + 1]]
        S = np.where(sim[:, J] > 0, sim[:, J], 0.0)
        if J.size > k:
            idx = np.argpartition(-S, k - 1, axis=1)[:, :k]; keep = np.zeros_like(S, bool); np.put_along_axis(keep, idx, True, 1); S = S * keep
        cnt = (S > 0).sum(1); den = S.sum(1); num = (S * (rj - mean[J])[None, :]).sum(1)
        ok = cnt >= min_k; Rh[e, ok] += num[ok] / den[ok]; Has[e] = cnt >= 1
    return Rh, Has

_SIMS = {}
def user_sims(s, tau, block=1000):
    """Thuật toán 2.2: chỉ giữ hàng độ tương đồng của người dùng đo (đủ cho mọi dự đoán cần tính)."""
    key = (id(s), tau)
    if key not in _SIMS:
        want = {int(u): e for e, u in enumerate(s.eu)}; rows = {}
        for u0, S in pearson_blocks(s.R, tau, block):
            for k in range(S.shape[0]):
                if u0 + k in want: rows[u0 + k] = S[k]
        _SIMS[key] = rows
    return _SIMS[key]

def m_userknn(s, k, tau, min_k=1):
    rows = user_sims(s, tau)
    Rh = np.tile(s.umean[s.eu][:, None], (1, s.nI)); Has = np.zeros((s.nE, s.nI), bool)
    for e, u in enumerate(s.eu):
        items, rhat, has = knn_scores_user(int(u), rows[int(u)], s.R, s.umean, k, min_k)
        Rh[e, items] = rhat; Has[e, items] = has
    return Rh, Has


# ============================================================ xếp hạng và độ đo (mục 2.6)
def rank_all(s, Rh, Has):
    """Khóa (2.8): có láng giềng, r_hat chưa cắt giảm dần, thứ tự business_id tăng dần; chỉ nhà hàng chưa đánh giá."""
    out = np.zeros((s.nE, s.nI), int); nc = np.zeros(s.nE, int)
    for e in range(s.nE):
        cand = np.flatnonzero(~s.Rseen[e]); nc[e] = cand.size
        o = np.lexsort((s.order[cand], -np.round(Rh[e, cand], 9), -Has[e, cand].astype(int)))
        out[e, :cand.size] = cand[o]
    return out, nc

def rerank(s, rank, nc, Rh, lam, M=100, anchor='mean', kind='minmax', sig=None, K=20, Has=None):
    D = s.dist(anchor); out = np.zeros((s.nE, K), int)
    for e in range(s.nE):
        m = nc[e] if M == 'all' else min(M, nc[e]); c = rank[e, :m]
        sel, _, _ = geo_rerank(Rh[e, c], D[e, c], K, lam, kind, None if sig is None else sig[e], s.order[c], None if Has is None else Has[e, c])
        out[e, :len(sel)] = c[sel]
    return out

def lists_of(rank, K=20): return rank[:, :K]

DISC = 1 / np.log2(np.arange(2, 22))

def per_user(s, L, K):
    hit = np.take_along_axis(s.Rel, L[:, :K], 1); nrel = s.Rel.sum(1)
    P = hit.sum(1) / K; Rc = hit.sum(1) / nrel
    dcg = (hit * DISC[:K]).sum(1); idcg = np.array([DISC[:min(K, n)].sum() for n in nrel])
    return P, Rc, dcg / idcg

def summarize(s, L, Dm, tag):
    row = {'model': tag}
    for K in KS:
        P, Rc, N = per_user(s, L, K)
        row.update({f'P@{K}': P.mean(), f'R@{K}': Rc.mean(), f'NDCG@{K}': N.mean(),
                    f'Cov@{K}': len(np.unique(L[:, :K])) / s.nI,
                    f'AvgPop@{K}': s.pop_all[L[:, :K]].mean(), f'AvgDist@{K}': np.take_along_axis(Dm, L[:, :K], 1).mean()})
    return row

def rmse_mae(s, Rh):
    e = s.E_pairs.u.map(s.pos_e).values; i = s.E_pairs.i.values
    err = np.clip(Rh[e, i], 1, 5) - s.E_pairs.stars.values
    return float(np.sqrt((err ** 2).mean())), float(np.abs(err).mean())


# ============================================================ chọn lambda (Thuật toán 2.6) và kiểm định (Thuật toán 2.7)
def select_lambda(s, rank, nc, Rh, B, delta=0.05, seed=0, **kw):
    Dm = s.dist('mean'); nd = []; ad = []
    for lam in LAMS:
        L = rerank(s, rank, nc, Rh, lam, **kw); nd.append(per_user(s, L, 10)[2]); ad.append(np.take_along_axis(Dm, L[:, :10], 1).mean())
    nd = np.array(nd); rng = np.random.default_rng(seed); idx = rng.integers(0, s.nE, (B, s.nE))
    boot = nd[:, idx].mean(2)                                     # (|lam|, B)
    Lq = np.quantile((boot[0] - boot) / boot[0], 0.95, axis=1)
    ok = [k for k in range(len(LAMS)) if Lq[k] < delta]
    best = sorted(ok, key=lambda k: (ad[k], -nd[k].mean(), -LAMS[k]))[0]
    tab = pd.DataFrame({'lambda': LAMS, 'NDCG@10': nd.mean(1), 'AvgDist@10': ad, 'Q95_loss': Lq, 'ok': [k in ok for k in range(len(LAMS))]})
    return LAMS[best], tab

def wilcoxon_drop0(d):
    d = d[d != 0]; n = len(d)
    if n < 10: return dict(n=n, z=np.nan, p=np.nan, r=np.nan)
    rk = pd.Series(np.abs(d)).rank().values
    T = min(rk[d > 0].sum(), rk[d < 0].sum())
    z = (T - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return dict(n=n, z=z, p=2 * norm.sf(abs(z)), r=z / np.sqrt(n))

def compare(a, b, B, seed=1):
    d = a - b; rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (B, len(d)))
    bm = d[idx].mean(1); out = dict(diff=d.mean(), ci_lo=np.quantile(bm, .025), ci_hi=np.quantile(bm, .975), **wilcoxon_drop0(d))
    try: out['p_pratt'] = wilcoxon(d, zero_method='pratt', method='approx').pvalue
    except Exception: out['p_pratt'] = np.nan
    return out

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, k in enumerate(o):
        run = max(run, min(1, (m - r) * p[k])); adj[k] = run
    return adj


# ============================================================ chương trình chính
def edge_params(sc, best_c, names):
    """Tham số nào của cấu hình tốt nhất nằm ở biên lưới (min hoặc max) và điểm Val còn khác láng giềng phía trong (không hòa)."""
    out = []
    for d, nm in enumerate(names):
        vals = sorted({c[d] for c in sc})
        if len(vals) < 2 or best_c[d] not in (vals[0], vals[-1]): continue
        inner = vals[1] if best_c[d] == vals[0] else vals[-2]
        nb = tuple(inner if i == d else x for i, x in enumerate(best_c))
        if nb in sc and abs(sc[best_c] - sc[nb]) < 1e-9: continue            # hòa với láng giềng: mô hình không nhạy tham số này
        out.append(f"{nm}={'min' if best_c[d] == vals[0] else 'max'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviews', required=True); ap.add_argument('--business', required=True)
    ap.add_argument('--out', default='results'); ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--nmin', type=int, default=5); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--ks', default='10,20,40,80'); ap.add_argument('--taus', default='2,5')
    ap.add_argument('--factors', default='20,50'); ap.add_argument('--regs', default='0.02,0.10'); ap.add_argument('--epochs', default='20,40')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    B = 200 if a.quick else 1000
    num = lambda s_, f=float: sorted({f(x) for x in s_.split(',')})
    knn_grid = [(10, 2), (20, 5)] if a.quick else list(itertools.product(num(a.ks, int), num(a.taus, int)))
    bmf_grid = [(20, .02, 20), (50, .1, 40)] if a.quick else list(itertools.product(num(a.factors, int), num(a.regs), num(a.epochs, int)))
    rv, bz, tr, va, te, T1, T2 = load_split(a)

    # ---------------- TN1: thống kê dữ liệu
    pd.DataFrame([dict(reviews=len(rv), users=rv.user_id.nunique(), restaurants=rv.business_id.nunique(), T1=T1, T2=T2,
                       train=len(tr), val=len(va), test=len(te))]).to_csv(f'{a.out}/TN1_data.csv', index=False)

    # ======================= GIAI ĐOẠN CHỌN: H = Train, đo trên Val
    log('GIAI ĐOẠN CHỌN (Train -> Val)')
    sv = Stage(tr, va, bz, a.nmin)
    def val_ndcg(Rh, Has):
        rk, nc = rank_all(sv, Rh, Has); return per_user(sv, rk, 10)[2].mean()
    tune = []; best = {}; scores = {}
    for name, grid, fn in (('UserKNN', knn_grid, lambda c: m_userknn(sv, *c)), ('ItemKNN', knn_grid, lambda c: m_itemknn(sv, *c)),
                           ('BiasedMF', bmf_grid, lambda c: m_bmf(sv, *c))):
        for c in grid:
            t0 = time.time(); v = val_ndcg(*fn(c)); tune.append(dict(model=name, config=str(c), val_ndcg10=v, sec=time.time() - t0))
            log(f'  {name} {c}: Val NDCG@10={v:.4f} ({time.time()-t0:.0f}s)')
            scores.setdefault(name, {})[c] = v
            if name not in best or v > best[name][1]: best[name] = (c, v)
    pd.DataFrame(tune).to_csv(f'{a.out}/TN2_tuning_val.csv', index=False)
    for name, (c, v) in best.items():
        names = ('k', 'tau') if name != 'BiasedMF' else ('factors', 'reg', 'epochs')
        edges = edge_params(scores[name], c, names)
        log(f'  chọn {name}: {c} Val NDCG@10={v:.4f}' + (f'  <-- NẰM Ở BIÊN LƯỚI theo {edges}: mở rộng lưới thêm một bước về phía đó rồi chạy lại' if edges else ''))
    # hạt giống chuẩn của Biased MF: Val NDCG@10 gần trung vị
    bm = best['BiasedMF'][0]; seed_val = {sd: val_ndcg(*m_bmf(sv, *bm, seed=sd)) for sd in SEEDS}
    ref_seed = sorted(seed_val, key=seed_val.get)[1]; log(f'  Biased MF Val theo hạt giống {seed_val} -> hạt giống chuẩn {ref_seed}')
    val_models = {'UserKNN': m_userknn(sv, *best['UserKNN'][0]), 'ItemKNN': m_itemknn(sv, *best['ItemKNN'][0]), 'BiasedMF': m_bmf(sv, *bm, seed=ref_seed)}
    base = max(val_models, key=lambda k: val_ndcg(*val_models[k])); log(f'  mô hình nền m* = {base}')
    Rh_b, Has_b = val_models[base]; rk_b, nc_b = rank_all(sv, Rh_b, Has_b)
    rk_p, nc_p = rank_all(sv, *m_mostpop(sv)); Rh_p = np.tile(sv.pop_pos, (sv.nE, 1))
    # biến thể TN4: (tên, kwargs)
    variants = {'main': {}, 'geomedian': dict(anchor='geomedian'), 'mindist': dict(anchor='mindist'), 'exp': dict(kind='exp'),
                'RR-50': dict(M=50), 'RR-all': dict(M='all')}
    sig_v = sv.sigma(); lam_sel = {}; lam_tabs = []
    for vn, kw in variants.items():
        kw2 = dict(kw);
        if kw2.get('kind') == 'exp': kw2['sig'] = sig_v
        lam, tab = select_lambda(sv, rk_b, nc_b, Rh_b, B, Has=Has_b, **kw2); lam_sel[vn] = lam; tab['variant'] = vn; lam_tabs.append(tab)
        log(f'  lambda* ({vn}) = {lam}')
    lam_P, tabP = select_lambda(sv, rk_p, nc_p, Rh_p, B); tabP['variant'] = 'PopNear'; lam_tabs.append(tabP); log(f'  lambda_P (PopNear) = {lam_P}')
    pd.concat(lam_tabs).to_csv(f'{a.out}/TN3_lambda_val.csv', index=False)
    json.dump(dict(best={k: [list(v[0]), v[1]] for k, v in best.items()}, ref_seed=ref_seed, base=base, lam=lam_sel, lam_P=lam_P),
              open(f'{a.out}/decisions_from_val.json', 'w'), indent=1)

    # ======================= GIAI ĐOẠN BÁO CÁO: H = Train ∪ Val, đo trên Test (không chọn lại gì ở giai đoạn này)
    log('GIAI ĐOẠN BÁO CÁO (Train+Val -> Test)')
    st = Stage(pd.concat([tr, va]), te, bz, a.nmin)
    cfg = {k: v[0] for k, v in best.items()}
    def seeds_models(): return {sd: m_bmf(st, *cfg['BiasedMF'], seed=sd) for sd in SEEDS}
    bmf_s = seeds_models()
    models = {'Random': m_random(st), 'GlobalMean': m_globalmean(st), 'MostPopular': m_mostpop(st), 'BaselineOnly': m_baseline(st),
              'UserKNN': m_userknn(st, *cfg['UserKNN']), 'ItemKNN': m_itemknn(st, *cfg['ItemKNN']), 'BiasedMF': bmf_s[ref_seed]}
    Dm = st.dist('mean'); rows = []; per = {}; ranks = {}
    for name, (Rh, Has) in models.items():
        rk, nc = rank_all(st, Rh, Has); ranks[name] = (rk, nc); L = lists_of(rk); r = summarize(st, L, Dm, name)
        if name not in ('Random', 'MostPopular'): r['RMSE'], r['MAE'] = rmse_mae(st, Rh)
        rows.append(r); per[name] = per_user(st, L, 10)[2]
    # độ lệch chuẩn của Biased MF qua 3 hạt giống
    sd_rows = [summarize(st, lists_of(rank_all(st, *bmf_s[sd])[0]), Dm, f'BiasedMF seed {sd}') for sd in SEEDS]
    pd.DataFrame(sd_rows).to_csv(f'{a.out}/TN2_BiasedMF_seeds_test.csv', index=False)
    # xếp hạng lại trên mô hình nền (biến thể 'main') và PopNear
    Rh_t, Has_t = models[base]; rk_t, nc_t = ranks[base]; rk_pt, nc_pt = ranks['MostPopular']; Rh_pt = models['MostPopular'][0]
    sig_t = st.sigma(); rr = {}
    chk = rerank(st, rk_t, nc_t, Rh_t, 1.0, Has=Has_t)
    same = (chk[:, :20] == rk_t[:, :20]).all(1).mean(); log(f'  KIỂM TRA: lambda=1 trùng mô hình nền ở {same:.1%} người dùng (phải = 100%)')
    if same < 1.0:
        bad = np.flatnonzero((chk[:, :20] != rk_t[:, :20]).any(1))[:3]
        for e in bad:
            j = int(np.flatnonzero(chk[e, :20] != rk_t[e, :20])[0]); ib, ir = rk_t[e, j], chk[e, j]
            log(f'  LỆCH user#{e} vị trí {j}: nền item {ib} (r_hat={Rh_t[e, ib]!r}, has={Has_t[e, ib]}) vs rerank item {ir} (r_hat={Rh_t[e, ir]!r}, has={Has_t[e, ir]}), nc={nc_t[e]}')
    assert same == 1.0, 'lambda=1 không trùng mô hình nền: lỗi xếp hạng lại'
    for vn, kw in variants.items():
        kw2 = dict(kw)
        if kw2.get('kind') == 'exp': kw2['sig'] = sig_t
        for lam, tag in ((1.0, f'{base}+rerank λ=1 [{vn}]'), (lam_sel[vn], f'{base}+rerank λ*={lam_sel[vn]} [{vn}]')):
            L = rerank(st, rk_t, nc_t, Rh_t, lam, Has=Has_t, **kw2); rows.append(summarize(st, L, Dm, tag)); rr[tag] = per_user(st, L, 10)[2]
            if vn == 'main': per[tag] = rr[tag]
    for lam in LAMS:      # TN3: toàn bộ lambda trên Test (chỉ để mô tả đường đánh đổi)
        L = rerank(st, rk_t, nc_t, Rh_t, lam, Has=Has_t); rows.append(summarize(st, L, Dm, f'{base}+rerank λ={lam}'))
    Lp = rerank(st, rk_pt, nc_pt, Rh_pt, lam_P); rows.append(summarize(st, Lp, Dm, f'PopNear λ_P={lam_P}')); per['PopNear'] = per_user(st, Lp, 10)[2]
    res = pd.DataFrame(rows); res.to_csv(f'{a.out}/TN2_TN3_TN4_test.csv', index=False)

    # ---------------- kiểm định (Thuật toán 2.7): họ so sánh NDCG@10, hiệu chỉnh Holm
    main_tag = f'{base}+rerank λ*={lam_sel["main"]} [main]'; one_tag = f'{base}+rerank λ=1 [main]'
    pairs = [(main_tag, one_tag, 'rerank vs λ=1 (RQ3/RQ4)'), (main_tag, 'PopNear', 'ΔCF: CF+rerank vs PopNear (RQ5)')] + \
            [(m, 'MostPopular', f'{m} vs MostPopular (RQ1)') for m in ('UserKNN', 'ItemKNN', 'BiasedMF', 'BaselineOnly')]
    per['MostPopular'] = per['MostPopular']; cmp_rows = []
    for x, y, lab in pairs:
        r = compare(per[x], per[y], B); r['comparison'] = lab; cmp_rows.append(r)
    cmp = pd.DataFrame(cmp_rows); cmp['p_holm'] = holm(cmp.p.fillna(1).values); cmp.to_csv(f'{a.out}/TN3_TN5_tests.csv', index=False)

    # ---------------- TN5/TN6: theo nhóm lịch sử (RQ2, RQ5)
    co = []
    for g, gn in enumerate(['1-4', '5-19', '>=20']):
        m = st.cohort == g
        for name in ('MostPopular', 'BaselineOnly', 'UserKNN', 'ItemKNN', 'BiasedMF', one_tag, main_tag, 'PopNear'):
            co.append(dict(cohort=gn, n_users=int(m.sum()), model=name, NDCG10=per[name][m].mean(),
                           CI_lo=np.quantile([per[name][m][np.random.default_rng(i).integers(0, m.sum(), m.sum())].mean() for i in range(B)], .025),
                           CI_hi=np.quantile([per[name][m][np.random.default_rng(i).integers(0, m.sum(), m.sum())].mean() for i in range(B)], .975)))
    pd.DataFrame(co).to_csv(f'{a.out}/TN5_TN6_cohorts.csv', index=False)

    # ---------------- TN7: dữ liệu để phân tích lỗi (người dùng NDCG@10 = 0 ở mô hình nền)
    z = pd.DataFrame({'user': [st.ts.to_raw_uid(int(u)) for u in st.eu], 'n_hist': st.R.getnnz(axis=1)[st.eu], 'n_rel': st.Rel.sum(1),
                      'ndcg10_base': per[base], 'ndcg10_rerank': per[main_tag], 'ndcg10_popnear': per['PopNear']})
    z.to_csv(f'{a.out}/TN7_per_user.csv', index=False)
    log('XONG. Các file kết quả trong', a.out)
    print(res[['model', 'NDCG@10', 'R@10', 'AvgDist@10', 'RMSE']].to_string(index=False))

if __name__ == '__main__':
    main()
