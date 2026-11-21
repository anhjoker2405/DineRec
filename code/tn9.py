"""TN9 (mục 3.12): xếp hạng lại trên tập ứng viên hợp nhất, công thức (3.2)–(3.3), Thuật toán 3.1.
C_u^∪ = M ứng viên đầu của ItemKNN ∪ M ứng viên có pop+ cao nhất (đúng tập ứng viên của PopNear), M = 100.
f = λ[α s_CF + (1 − α) s_pop] + (1 − λ) g, cả ba chuẩn hóa min–max trên C_u^∪; ứng viên không có láng giềng nhận s_CF = 0
(cùng quy ước với Thuật toán 2.5). Khi hòa: điểm sở thích giảm dần, rồi nhà hàng có láng giềng trước, rồi r_hat giảm dần,
rồi business_id, tức cùng thứ tự phá hòa với Thuật toán 2.5; nhờ vậy α = 1, λ = 1 trùng đúng danh sách của ItemKNN.
Mỗi α: chọn λ_α trên Val bằng đúng Thuật toán 2.6 (δ = 0,05, B = 1.000, hạt giống 0, khoảng cách trung bình nhỏ nhất trong các λ đạt).
α* = α có NDCG@10 trên Val tại λ_α cao nhất, hòa thì α nhỏ hơn. Test chỉ để báo cáo; so với PopNear bằng Thuật toán 2.7.
TN9 là thí nghiệm bổ sung, thiết kế sau khi đã có kết quả mục 3.6, nên được báo cáo như phân tích thăm dò (ngoài họ Holm).
Dùng (trong thư mục code): python tn9.py indy_reviews.csv indy_business.csv results3 results3/tn9.json"""
import sys, json, argparse, time
rev, biz, res, outp = sys.argv[1:5]
import numpy as np, pandas as pd
import run_experiments as X

t0 = time.time()
M, K, DELTA, B = 100, 10, 0.05, 1000
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
dec = json.load(open(f'{res}/decisions_from_val.json'))
cfg = tuple(dec['best']['ItemKNN'][0]); lam_star = dec['lam']['main']; lamP = dec['lam_P']
a = argparse.Namespace(reviews=rev, business=biz, start='2018-01-01', nmin=5)
rv, bz, tr, va, te, T1, T2 = X.load_split(a)


def mm(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)


def prepare(s):
    """Tập ứng viên hợp nhất và ba điểm đã chuẩn hóa cho từng người dùng đo."""
    Rh, Has = X.m_itemknn(s, *cfg); rk, nc = X.rank_all(s, Rh, Has)
    Rp = X.m_mostpop(s); rkp, ncp = X.rank_all(s, *Rp); Dm = s.dist('mean')
    P = []
    for e in range(s.nE):
        ccf = rk[e, :min(M, nc[e])]; cpop = rkp[e, :min(M, ncp[e])]
        U = np.unique(np.r_[ccf, cpop])
        r = np.round(Rh[e, U], 9); h = Has[e, U]
        if h.any():
            sc = np.where(h, mm(np.where(h, r, r[h].min())), 0.0) if r[h].max() > r[h].min() else np.where(h, 1.0, 0.0)
        else:
            sc = np.zeros(U.size)
        P.append(dict(U=U, scf=sc, spop=mm(s.pop_pos[U]), g=1 - mm(Dm[e, U]), d=Dm[e, U], ord=s.order[U], has=h, r=r,
                      n_cf=ccf.size, n_pop=cpop.size))
    base = dict(Rh=Rh, Has=Has, rk=rk, nc=nc, rkp=rkp, ncp=ncp, Rp=Rp[0], Dm=Dm)
    return P, base


def lists(P, alpha, lam):
    L = np.zeros((len(P), K), int)
    for e, p in enumerate(P):
        pref = alpha * p['scf'] + (1 - alpha) * p['spop']; f = lam * pref + (1 - lam) * p['g']
        o = np.lexsort((p['ord'], -p['r'], -p['has'].astype(int), -np.round(pref, 12), -np.round(f, 12)))[:K]
        L[e] = p['U'][o]
    return L


def evaluate(s, P, alpha, lam):
    L = lists(P, alpha, lam); Pk, Rk, N = X.per_user(s, L, K); d = np.take_along_axis(s.dist('mean'), L, 1)
    return L, N, dict(ndcg=float(N.mean()), recall=float(Rk.mean()), avgdist=float(d.mean()), median=float(np.median(d)), le5=float((d <= 5).mean()))


OUT = dict(M=M, K=K, alphas=ALPHAS, lams=X.LAMS)
# ---------------- Val
sv = X.Stage(tr, va, bz, 5)
Pv, bv = prepare(sv)
OUT['val_union_size'] = dict(mean=float(np.mean([p['U'].size for p in Pv])), min=int(min(p['U'].size for p in Pv)), max=int(max(p['U'].size for p in Pv)))
idx = np.random.default_rng(0).integers(0, sv.nE, (B, sv.nE))
val_rows, sel = [], {}
for al in ALPHAS:
    nd, st_ = [], []
    for lam in X.LAMS:
        _, N, m_ = evaluate(sv, Pv, al, lam); nd.append(N); st_.append(m_)
    nd = np.array(nd); boot = nd[:, idx].mean(2); q = np.quantile((boot[0] - boot) / boot[0], 0.95, axis=1)
    ok = [k for k in range(len(X.LAMS)) if q[k] < DELTA]
    kb = sorted(ok, key=lambda k: (st_[k]['avgdist'], -nd[k].mean(), -X.LAMS[k]))[0]
    sel[al] = dict(lam=X.LAMS[kb], **st_[kb])
    for k, lam in enumerate(X.LAMS): val_rows.append(dict(alpha=al, lam=lam, Q95=float(q[k]), ok=k in ok, **st_[k]))
alpha_star = sorted(ALPHAS, key=lambda al: (-round(sel[al]['ndcg'], 12), al))[0]; lam_a = sel[alpha_star]['lam']
chk_v = [r['ndcg'] for r in val_rows if r['alpha'] == 1.0 and r['lam'] == 1.0][0]
Lb = X.rank_all(sv, bv['Rh'], bv['Has'])[0][:, :K]; nb = float(X.per_user(sv, Lb, K)[2].mean())
assert abs(chk_v - nb) < 1e-12, (chk_v, nb)
OUT['check_alpha1_lam1_equals_itemknn_val'] = nb
OUT['val_grid'] = val_rows; OUT['val_selected_per_alpha'] = {str(k): v for k, v in sel.items()}
OUT['alpha_star'] = alpha_star; OUT['lam_alpha_star'] = lam_a
# đối chứng trên Val: PopNear và RR-100 λ* (cùng cách tính với run_experiments)
Lp = X.rerank(sv, bv['rkp'], bv['ncp'], bv['Rp'], lamP)[:, :K]; Lr = X.rerank(sv, bv['rk'], bv['nc'], bv['Rh'], lam_star, Has=bv['Has'])[:, :K]
OUT['val_ref'] = dict(popnear=float(X.per_user(sv, Lp, K)[2].mean()), rr100=float(X.per_user(sv, Lr, K)[2].mean()))
OUT['sec_val'] = time.time() - t0

# ---------------- Test (chỉ báo cáo)
st = X.Stage(pd.concat([tr, va]), te, bz, 5)
Pt, bt = prepare(st)
Dm = st.dist('mean'); nrel = st.Rel.sum(1)
h = np.array([st.Rel[e, p['U']].sum() for e, p in enumerate(Pt)])
OUT['test_union'] = dict(mean_size=float(np.mean([p['U'].size for p in Pt])), share_users_no_rel=float((h == 0).mean()),
                         recall_at_union=float((h / nrel).mean()), ceiling_recall10=float((np.minimum(h, K) / nrel).mean()))
_, Nchk, _ = evaluate(st, Pt, 1.0, 1.0)                                    # tự kiểm: α = 1, λ = 1 trùng ItemKNN trên Test
nb_t = float(X.per_user(st, X.rank_all(st, bt['Rh'], bt['Has'])[0][:, :K], K)[2].mean()); assert abs(Nchk.mean() - nb_t) < 1e-12
OUT['check_alpha1_lam1_equals_itemknn_test'] = nb_t
L9, N9, m9 = evaluate(st, Pt, alpha_star, lam_a)
Lp = X.rerank(st, bt['rkp'], bt['ncp'], bt['Rp'], lamP)[:, :K]; NP = X.per_user(st, Lp, K)[2]
Lr = X.rerank(st, bt['rk'], bt['nc'], bt['Rh'], lam_star, Has=bt['Has'])[:, :K]; NR = X.per_user(st, Lr, K)[2]
cmp = lambda a_, b_: {k: float(v) for k, v in X.compare(a_, b_, B).items()}
# kiểm tra: PopNear và RR-100 trùng tệp kết quả đã lưu
tt = pd.read_csv(f'{res}/TN2_TN3_TN4_test.csv').set_index('model')
OUT['check_saved'] = dict(popnear=float(abs(tt.loc[f'PopNear λ_P={lamP}', 'NDCG@10'] - NP.mean())),
                          rr100=float(abs(tt.loc[f'ItemKNN+rerank λ*={lam_star} [main]', 'NDCG@10'] - NR.mean())))
OUT['test'] = dict(tn9=m9, popnear=float(NP.mean()), rr100=float(NR.mean()),
                   tn9_vs_popnear=cmp(N9, NP), tn9_vs_rr100=cmp(N9, NR), share_nonzero=float((N9 > 0).mean()))
# mô tả: mọi α tại λ_α của nó trên Test (không dùng để chọn)
OUT['test_all_alpha'] = {}
for al in ALPHAS:
    _, N_, m_ = evaluate(st, Pt, al, sel[al]['lam']); OUT['test_all_alpha'][str(al)] = dict(lam=sel[al]['lam'], **m_, vs_popnear=cmp(N_, NP))
OUT['sec'] = time.time() - t0
json.dump(OUT, open(outp, 'w'), indent=1, ensure_ascii=False)

print('union Val', OUT['val_union_size'], 'Test', OUT['test_union'])
print(pd.DataFrame(val_rows).round(4).to_string(index=False))
print('selected per alpha (Val):'); print(pd.DataFrame(sel).T.round(4))
print('alpha*', alpha_star, 'lam_alpha*', lam_a, 'val ref', OUT['val_ref'])
print('check saved', OUT['check_saved'])
print('TEST tn9', m9); print('vs PopNear', OUT['test']['tn9_vs_popnear']); print('vs RR100', OUT['test']['tn9_vs_rr100'])
for k, v in OUT['test_all_alpha'].items(): print(' test alpha', k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'vs_popnear'}, 'diff', round(v['vs_popnear']['diff'], 4), [round(v['vs_popnear']['ci_lo'], 4), round(v['vs_popnear']['ci_hi'], 4)])
print('sec', OUT['sec'])
