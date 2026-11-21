"""Số liệu và hình bổ sung cho Chương 3 (không chọn lại gì: mọi quyết định đọc từ decisions_from_val.json).
Dùng: python ch3_analysis.py REVIEWS BUSINESS RESULTS_DIR OUT_DIR"""
import sys, json, argparse, os
import numpy as np, pandas as pd
sys.path.insert(0, '.')
import run_experiments as X
from dinerec_core import haversine, project
import matplotlib
matplotlib.use('Agg')
import vn_format  # dấu phẩy thập phân cho mọi hình (quy ước tiếng Việt)
import matplotlib.pyplot as plt

rev, biz, res, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
C = dict(base='#1f5a96', rr='#d9822b', pop='#7a7a7a', pn='#2e8b57', ref='#8e44ad')
dec = json.load(open(f'{res}/decisions_from_val.json'))
cfg = {k: tuple(v[0]) for k, v in dec['best'].items()}
NUM = {}

a = argparse.Namespace(reviews=rev, business=biz, start='2018-01-01', nmin=5)
rv, bz, tr, va, te, T1, T2 = X.load_split(a)
NUM['T1'], NUM['T2'] = str(T1), str(T2)

# ============================================================ Bảng 3.3: thống kê các tập
def filt(H, nmin=5):
    c = H.groupby('business_id').user_id.nunique(); return H[H.business_id.isin(c[c >= nmin].index)]
def stats(df, name):
    n = len(df); U = df.user_id.nunique(); I = df.business_id.nunique(); g = df.groupby('user_id').size()
    return dict(tap=name, tuong_tac=n, nguoi_dung=U, nha_hang=I, do_thua=1 - n / (U * I), tb_moi_nguoi=n / U,
                ty_le_1_4=float((g <= 4).mean()), sao_tb=df.stars.mean(), ty_le_tich_cuc=float((df.stars >= 4).mean()))
Hv, Ht = filt(tr), filt(pd.concat([tr, va]))
rows = [stats(rv, 'Toàn cửa sổ'), stats(tr, 'Train'), stats(va, 'Val'), stats(te, 'Test'), stats(Hv, 'H_val'), stats(Ht, 'H_test')]
def eval_breakdown(Hraw, Hf, E, name):
    pos = E[E.stars >= 4]; uH = set(Hf.user_id); iraw = set(Hraw.business_id); icat = set(Hf.business_id)
    seen = set(zip(Hf.user_id, Hf.business_id))
    newu = ~pos.user_id.isin(uH); newi = ~pos.business_id.isin(iraw); below = pos.business_id.isin(iraw) & ~pos.business_id.isin(icat)
    rep = np.array([(u, b) in seen for u, b in zip(pos.user_id, pos.business_id)])
    kept = ~newu & pos.business_id.isin(icat) & ~rep
    only_eval = len(set(E.business_id) - iraw) / E.business_id.nunique()
    return dict(tap=name, danh_gia=len(E), tich_cuc=len(pos), user_moi=int(newu.sum()), nha_hang_moi=int((~newu & newi).sum()),
                duoi_nmin=int((~newu & below).sum()), lap_lai=int((~newu & rep).sum()), giu_lai=int(kept.sum()),
                nguoi_dung_do=pos[kept].user_id.nunique(), ty_le_nha_hang_chi_o_tap_do=only_eval)
br = [eval_breakdown(tr, Hv, va, 'Val'), eval_breakdown(pd.concat([tr, va]), Ht, te, 'Test')]
pd.DataFrame(rows).to_csv(f'{out}/B33_stats.csv', index=False); pd.DataFrame(br).to_csv(f'{out}/B33_eval_breakdown.csv', index=False)
for H, nm in ((Hv, 'H_val'), (Ht, 'H_test')):
    cu = H.groupby('business_id').size().values.astype(float); NUM[f'sumUi2_{nm}'] = float((cu ** 2).sum()); NUM[f'U2_{nm}'] = float(H.user_id.nunique()) ** 2

# ============================================================ Hình 3.1-3.4: dữ liệu
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
for k, (col, lab) in enumerate((('user_id', 'người dùng'), ('business_id', 'nhà hàng'))):
    c = rv.groupby(col).size(); f = c.value_counts().sort_index()
    ax[k].loglog(f.index, f.values, 'o', ms=2.5, color=C['base'])
    ax[k].set_xlabel(f'Số đánh giá mỗi {lab}'); ax[k].set_ylabel(f'Số {lab}')
    ax[k].set_title(f'({"ab"[k]}) Theo {lab}: trung vị {int(c.median())}, tối đa {vn_format.ng(c.max())}', fontsize=8.5)
    NUM[f'median_per_{col}'] = float(c.median()); NUM[f'max_per_{col}'] = int(c.max()); NUM[f'mean_per_{col}'] = float(c.mean())
fig.tight_layout(); fig.savefig(f'{out}/fig31_loglog.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 2.5)); w = .27
for k, (df, nm, col) in enumerate(((tr, 'Train', C['base']), (va, 'Val', C['rr']), (te, 'Test', C['pn']))):
    p = df.stars.value_counts(normalize=True).reindex(range(1, 6), fill_value=0) * 100
    ax.bar(np.arange(1, 6) + (k - 1) * w, p.values, w, label=nm, color=col)
    NUM[f'stars_{nm}'] = [round(x, 1) for x in p.values]
ax.set_xlabel('Số sao'); ax.set_ylabel('Tỷ lệ đánh giá (%)'); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f'{out}/fig32_stars.png'); plt.close(fig)

m = rv.set_index('date').resample('MS').size()
mf = m[m.index + pd.offsets.MonthEnd(0) <= rv.date.max().normalize()]   # chỉ vẽ tháng đủ dữ liệu: tháng 1/2022 chỉ đến 19/01 nên bỏ
fig, ax = plt.subplots(figsize=(6.6, 2.5)); ax.plot(mf.index, mf.values, color=C['base'], lw=1.3)
for t, lab in ((T1, 'T₁'), (T2, 'T₂')):
    ax.axvline(t, color=C['rr'], ls='--', lw=1); ax.text(t, m.max() * .97, ' ' + lab, color=C['rr'], va='top')
# nhà hàng ngừng phục vụ tại chỗ từ sắc lệnh 20-04 của bang Indiana (16/03/2020) đến khi Marion County cho phục vụ trong nhà (01/06/2020)
ax.axvspan(pd.Timestamp('2020-03-16'), pd.Timestamp('2020-06-01'), color='#cccccc', alpha=.5, lw=0)
ax.text(pd.Timestamp('2020-04-24'), m.max() * .98, 'ngừng phục vụ\ntại chỗ', ha='center', va='top', fontsize=7)
ax.text(1, -0.2, f'Tháng {rv.date.max():%m/%Y} chỉ có dữ liệu đến ngày {rv.date.max():%d/%m} nên không vẽ.', transform=ax.transAxes, ha='right', va='top', fontsize=6.5, color='#555555')
ax.set_ylabel('Số đánh giá mỗi tháng'); fig.tight_layout(); fig.savefig(f'{out}/fig33_monthly.png'); plt.close(fig)
NUM['monthly_2019_mean'] = float(m['2019'].mean()); NUM['monthly_2020_04'] = int(m['2020-04'].iloc[0]); NUM['monthly_2021_mean'] = float(m['2021'].mean())
NUM['last_review'] = str(rv.date.max())

b = bz[bz.business_id.isin(set(rv.business_id))].copy(); b['n'] = b.business_id.map(rv.groupby('business_id').size())
b['cat'] = b.business_id.isin(set(Ht.business_id))
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.scatter(b.longitude[~b.cat], b.latitude[~b.cat], s=4, c='#bbbbbb', label=f'Ngoài danh mục ({vn_format.ng((~b.cat).sum())})', lw=0)
ax.scatter(b.longitude[b.cat], b.latitude[b.cat], s=4 + 40 * b.n[b.cat] / b.n.max(), c=C['base'], alpha=.6, lw=0, label=f'Trong danh mục $I_H$ ({vn_format.ng(b.cat.sum())})')
ax.set_aspect(1 / np.cos(np.radians(b.latitude.mean()))); ax.set_xlabel('Kinh độ'); ax.set_ylabel('Vĩ độ'); ax.legend(frameon=False, fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig(f'{out}/fig34_map.png'); plt.close(fig)
la0, lo0 = b.latitude.median(), b.longitude.median()
NUM['radius_p95_km'] = float(np.quantile(haversine(la0, lo0, b.latitude.values, b.longitude.values), .95))
NUM['radius_max_km'] = float(haversine(la0, lo0, b.latitude.values, b.longitude.values).max())

# ============================================================ giai đoạn báo cáo: mô hình với quyết định đã chốt
st = X.Stage(pd.concat([tr, va]), te, bz, 5)
bmf = {sd: X.m_bmf(st, *cfg['BiasedMF'], seed=sd) for sd in X.SEEDS}
models = {'Random': X.m_random(st), 'GlobalMean': X.m_globalmean(st), 'MostPopular': X.m_mostpop(st), 'BaselineOnly': X.m_baseline(st),
          'UserKNN': X.m_userknn(st, *cfg['UserKNN']), 'ItemKNN': X.m_itemknn(st, *cfg['ItemKNN']), 'BiasedMF': bmf[dec['ref_seed']]}
base = dec['base']; lam = dec['lam']; lamP = dec['lam_P']
Dm = st.dist('mean'); ranks = {k: X.rank_all(st, *v) for k, v in models.items()}
Rh_b, Has_b = models[base]; rk_b, nc_b = ranks[base]
L = {k: X.lists_of(ranks[k][0]) for k in models}
L[f'{base} λ=1'] = L[base]
L[f'RR-100 λ*={lam["main"]}'] = X.rerank(st, rk_b, nc_b, Rh_b, lam['main'], Has=Has_b)
L['RR-100 λ=0'] = X.rerank(st, rk_b, nc_b, Rh_b, 0.0, Has=Has_b)
L[f'RR-toàn bộ λ*={lam["RR-all"]}'] = X.rerank(st, rk_b, nc_b, Rh_b, lam['RR-all'], M='all', Has=Has_b)
L['RR-toàn bộ λ=0'] = X.rerank(st, rk_b, nc_b, Rh_b, 0.0, M='all', Has=Has_b)
rk_p, nc_p = ranks['MostPopular']
L[f'PopNear λ_P={lamP}'] = X.rerank(st, rk_p, nc_p, models['MostPopular'][0], lamP)
rng = np.random.default_rng(2); IDX = rng.integers(0, st.nE, (1000, st.nE))
def ci(x): bm = x[IDX].mean(1); return float(x.mean()), float(np.quantile(bm, .025)), float(np.quantile(bm, .975))
tab = []; PU = {}
for nm, Lm in L.items():
    row = dict(model=nm)
    for K in (5, 10, 20):
        P, Rc, N = X.per_user(st, Lm, K); ad = np.take_along_axis(Dm, Lm[:, :K], 1).mean(1)
        for mn, v in (('P', P), ('R', Rc), ('NDCG', N)):
            m_, lo, hi = ci(v); row[f'{mn}@{K}'] = m_; row[f'{mn}@{K}_lo'] = lo; row[f'{mn}@{K}_hi'] = hi
        row[f'AvgDist@{K}'] = ad.mean(); row[f'MedDist@{K}'] = float(np.median(ad))
        row[f'Cov@{K}'] = len(np.unique(Lm[:, :K])) / st.nI; row[f'AvgPop@{K}'] = st.pop_all[Lm[:, :K]].mean()
        if K == 10: PU[nm] = dict(N=N, R=Rc, D=ad)
    tab.append(row)
T = pd.DataFrame(tab); T.to_csv(f'{out}/B36_metrics_ci.csv', index=False)
sdr = []
for sd in X.SEEDS:
    r_ = X.rmse_mae(st, bmf[sd][0]); Lsd = X.lists_of(X.rank_all(st, *bmf[sd])[0])
    sdr.append(dict(seed=sd, RMSE=r_[0], MAE=r_[1], **{f'{mn}@{K}': v.mean() for K in (5, 10, 20) for mn, v in zip(('P', 'R', 'NDCG'), X.per_user(st, Lsd, K))}))
pd.DataFrame(sdr).to_csv(f'{out}/B35_bmf_seeds.csv', index=False)
NUM['n_eval_test'] = int(st.nE); NUM['n_rmse_pairs'] = int(len(st.E_pairs))

# ---- tỷ lệ dự phòng (mục 2.4.2) và đồng hạng do cắt điểm (mục 2.5.2)
e_ = st.E_pairs.u.map(st.pos_e).values; i_ = st.E_pairs.i.values
for nm in ('UserKNN', 'ItemKNN'):
    Rh, Has = models[nm]
    NUM[f'fallback_rmse_{nm}'] = float(1 - Has[e_, i_].mean())
    NUM[f'fallback_top10_{nm}'] = float(1 - np.take_along_axis(Has, L[nm][:, :10], 1).mean())
    NUM[f'users_no_neighbour_top10_{nm}'] = float((~np.take_along_axis(Has, L[nm][:, :10], 1)).all(1).mean())
te_all = te.copy(); uH = set(st.uid); iH = set(st.items)
NUM['test_ratings_total'] = int(len(te)); NUM['test_ratings_user_unknown'] = int((~te.user_id.isin(uH)).sum())
NUM['test_ratings_item_unknown'] = int((te.user_id.isin(uH) & ~te.business_id.isin(iH)).sum())
for nm in ('UserKNN', 'ItemKNN', 'BiasedMF', 'BaselineOnly'):
    Rh = models[nm][0]; top = np.take_along_axis(Rh, ranks[nm][0][:, :100], 1)
    NUM[f'clip_ties_users_{nm}'] = float(((top >= 5).sum(1) >= 2).mean()); NUM[f'clip_max_{nm}'] = int((top >= 5).sum(1).max())

# ---- trần (2.19) với M = 50, 100, toàn bộ trên mô hình nền
DISC = 1 / np.log2(np.arange(2, 3000)); nrel = st.Rel.sum(1)
ceil = []
for M in (50, 100, 'all'):
    h = np.array([st.Rel[e, rk_b[e, :(nc_b[e] if M == 'all' else min(M, nc_b[e]))]].sum() for e in range(st.nE)])
    for K in (10,):
        hk = np.minimum(h, K); rc = hk / nrel
        nd = np.array([DISC[:hk[e]].sum() / DISC[:min(K, nrel[e])].sum() for e in range(st.nE)])
        ceil.append(dict(M=str(M), K=K, recall_ceiling=rc.mean(), ndcg_ceiling=nd.mean(), share_users_no_relevant_in_CM=float((h == 0).mean())))
pd.DataFrame(ceil).to_csv(f'{out}/B311_ceiling.csv', index=False)

# ---- không thua kém của lambda* trên Test (cùng tiêu chí Thuật toán 2.6)
nb1 = PU[f'{base} λ=1']['N']; nbs = PU[f'RR-100 λ*={lam["main"]}']['N']
bt1, bts = nb1[IDX].mean(1), nbs[IDX].mean(1); Ltest = (bt1 - bts) / bt1
NUM['test_L_point'] = float((nb1.mean() - nbs.mean()) / nb1.mean()); NUM['test_L_q95'] = float(np.quantile(Ltest, .95))

# ---- khoảng cách tham chiếu và phân bố khoảng cách của danh sách (Hình 3.5, 3.8)
ref = Dm[st.Rel]; allc = np.concatenate([Dm[e][~st.Rseen[e]] for e in range(st.nE)])
NUM['ref_median_km'] = float(np.median(ref)); NUM['ref_mean_km'] = float(ref.mean()); NUM['ref_p75_km'] = float(np.quantile(ref, .75))
NUM['cand_median_km'] = float(np.median(allc)); NUM['ref_share_le5km'] = float((ref <= 5).mean())
NUM['ref_user_median_of_mean_km'] = float(np.median([Dm[e][st.Rel[e]].mean() for e in range(st.nE)]))
fig, ax = plt.subplots(figsize=(5.2, 2.6)); bins = np.linspace(0, 40, 41)
ax.hist(allc, bins, density=True, color='#cccccc', label='Mọi ứng viên chưa đánh giá')
ax.hist(ref, bins, density=True, histtype='step', lw=1.6, color=C['ref'], label='Nhà hàng liên quan (Test)')
ax.set_xlabel('Khoảng cách đến vị trí đại diện (km)'); ax.set_ylabel('Mật độ'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{out}/fig35_refdist.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for nm, lab, col, ls in ((f'{base} λ=1', 'λ = 1 (ItemKNN)', C['base'], '-'), (f'RR-100 λ*={lam["main"]}', f'λ* = {lam["main"]}', C['rr'], '-'),
                         ('RR-100 λ=0', 'λ = 0', C['pop'], '--')):
    d = np.take_along_axis(Dm, L[nm][:, :10], 1).ravel(); x = np.sort(d); ax.plot(x, np.arange(1, x.size + 1) / x.size, color=col, ls=ls, label=lab)
x = np.sort(ref); ax.plot(x, np.arange(1, x.size + 1) / x.size, color=C['ref'], lw=1.8, label='Tham chiếu (liên quan)')
ax.set_xlim(0, 30); ax.set_xlabel('Khoảng cách đến vị trí đại diện (km)'); ax.set_ylabel('Tỷ lệ tích lũy'); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f'{out}/fig38_ecdf.png')
fig.set_size_inches(3.5, 2.7); fig.tight_layout(); fig.savefig(f'{out}/fig38_ecdf_motcot.png'); plt.close(fig)  # bản một cột cho bài báo SSRC

# ---- Hình 3.6 NDCG@10 có thanh sai số
sel = ['Random', 'GlobalMean', 'MostPopular', 'BaselineOnly', 'UserKNN', 'ItemKNN', 'BiasedMF', f'RR-100 λ*={lam["main"]}', f'PopNear λ_P={lamP}']
Ts = T.set_index('model').loc[sel]
fig, ax = plt.subplots(figsize=(6.4, 2.8)); cols = [C['pop']] * 4 + [C['base']] * 3 + [C['rr'], C['pn']]
ax.bar(range(len(sel)), Ts['NDCG@10'], color=cols, yerr=[Ts['NDCG@10'] - Ts['NDCG@10_lo'], Ts['NDCG@10_hi'] - Ts['NDCG@10']], capsize=2.5, error_kw=dict(lw=.8))
ax.set_xticks(range(len(sel))); ax.set_xticklabels(['Random', 'GlobalMean', 'MostPop.', 'Baseline', 'UserKNN', 'ItemKNN', 'BiasedMF', 'ItemKNN\n+RR λ*', 'PopNear'], fontsize=7.5)
ax.set_ylabel('NDCG@10 (Test)'); fig.tight_layout(); fig.savefig(f'{out}/fig36_ndcg.png'); plt.close(fig)

# ---- nhóm người dùng (Bảng 3.9, 3.14; Hình 3.7)
coh = []; names = ['MostPopular', 'BaselineOnly', 'UserKNN', 'ItemKNN', 'BiasedMF', f'RR-100 λ*={lam["main"]}', f'PopNear λ_P={lamP}']
for c, lab in enumerate(('1–4', '5–19', '≥20')):
    msk = st.cohort == c; n = int(msk.sum()); ix = np.random.default_rng(3).integers(0, n, (1000, n))
    for nm in names:
        r = dict(cohort=lab, n=n, model=nm)
        for k_, v in (('NDCG@10', PU[nm]['N'][msk]), ('R@10', PU[nm]['R'][msk])):
            bm = v[ix].mean(1); r[k_] = v.mean(); r[k_ + '_lo'] = np.quantile(bm, .025); r[k_ + '_hi'] = np.quantile(bm, .975)
        r['AvgDist@10'] = PU[nm]['D'][msk].mean(); coh.append(r)
CO = pd.DataFrame(coh); CO.to_csv(f'{out}/B39_cohorts.csv', index=False)
fig, ax = plt.subplots(figsize=(6.4, 2.8)); w = .12; labs = ['MostPop.', 'Baseline', 'UserKNN', 'ItemKNN', 'BiasedMF', 'ItemKNN+RR λ*', 'PopNear']
colz = [C['pop'], '#b0b0b0', '#7fa7cf', C['base'], '#0d2f52', C['rr'], C['pn']]
for k, nm in enumerate(names):
    s_ = CO[CO.model == nm]; ax.bar(np.arange(3) + (k - 3) * w, s_['NDCG@10'], w, color=colz[k], label=labs[k],
                                   yerr=[s_['NDCG@10'] - s_['NDCG@10_lo'], s_['NDCG@10_hi'] - s_['NDCG@10']], capsize=1.5, error_kw=dict(lw=.6))
ax.set_xticks(range(3)); ax.set_xticklabels([f'{l} tương tác (n = {n})' for l, n in zip(('1–4', '5–19', '≥20'), CO.groupby('cohort', sort=False).n.first())])
ax.set_ylabel('NDCG@10 (Test)'); ax.legend(frameon=False, fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.22)); fig.tight_layout(); fig.savefig(f'{out}/fig37_cohort.png'); plt.close(fig)

# ---- Hình 3.9 đường đánh đổi (Val, biến thể chính) và Test
lv = pd.read_csv(f'{res}/TN3_lambda_val.csv'); lv = lv[lv.variant == 'main']
tt = pd.read_csv(f'{res}/TN2_TN3_TN4_test.csv'); tt = tt[tt.model.str.match(r'^\w+\+rerank λ=\d\.\d$')].copy(); tt['lambda'] = tt.model.str.extract(r'λ=(\d\.\d)').astype(float)
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.plot(lv['AvgDist@10'], lv['NDCG@10'], 'o-', color=C['base'], ms=4, label='Val (chọn λ)')
for _, r in lv.iterrows():
    ax.annotate(f'{r["lambda"]:.1f}', (r['AvgDist@10'], r['NDCG@10']), fontsize=6.5, xytext=(3, 3), textcoords='offset points', color='#c0392b' if not r.ok else 'black')
ax.plot(tt['AvgDist@10'], tt['NDCG@10'], 's--', color=C['rr'], ms=3.5, label='Test (chỉ báo cáo)')
k_ = lv['lambda'].sub(lam['main']).abs().idxmin(); ax.plot(lv.loc[k_, 'AvgDist@10'], lv.loc[k_, 'NDCG@10'], '*', ms=13, color=C['rr'], mec='black', label=f'λ* = {lam["main"]}')
ax.set_xlabel('AvgDistance@10 (km)'); ax.set_ylabel('NDCG@10'); ax.legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(f'{out}/fig39_tradeoff.png')
for t_ in ax.texts:  # bản một cột: bớt nhãn λ để không chồng nhau
    if t_.get_text() in ('0,3', '0,4', '0,6', '0,7', '0,8'): t_.set_visible(False)
fig.set_size_inches(3.5, 2.7); fig.tight_layout(); fig.savefig(f'{out}/fig39_tradeoff_motcot.png'); plt.close(fig)  # bản một cột cho bài báo SSRC

# ---- TN7: ba người dùng tiêu biểu, chọn theo tiêu chí cố định
ev = pd.DataFrame({'e': range(st.nE), 'u': st.eu, 'nh': st.R.getnnz(axis=1)[st.eu], 'mean': st.umean[st.eu], 'nrel': nrel})
def sep(e):
    u = st.eu[e]; J = st.R.indices[st.R.indptr[u]:st.R.indptr[u + 1]]
    if J.size < 6: return 0.0
    P = project(st.lat[J], st.lon[J], st.lat[J].mean(), st.lon[J].mean()); c0, c1 = P[np.argmin(P[:, 0])], P[np.argmax(P[:, 0])]
    for _ in range(20):
        lab = np.linalg.norm(P - c0, axis=1) > np.linalg.norm(P - c1, axis=1)
        if lab.all() or (~lab).all(): return 0.0
        c0, c1 = P[~lab].mean(0), P[lab].mean(0)
    return float(np.linalg.norm(c0 - c1)) if min(lab.sum(), (~lab).sum()) >= 3 else 0.0
ev['sep'] = [sep(e) for e in ev.e]
cases = {'A: hai khu vực': int(ev.sort_values(['sep', 'u'], ascending=[False, True]).e.iloc[0]),
         'B: ít lịch sử': int(ev[(ev.nh <= 4) & (ev.nrel >= 3)].sort_values(['nrel', 'u'], ascending=[False, True]).e.iloc[0]),
         'C: thiên lệch điểm cao': int(ev[(ev.nh >= 10) & (ev['mean'] >= 4.8)].sort_values(['nh', 'u'], ascending=[False, True]).e.iloc[0])}
Lb, Ls = L[f'{base} λ=1'], L[f'RR-100 λ*={lam["main"]}']
fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.3)); crow = []
for ax, (nm, e) in zip(axs, cases.items()):
    u = st.eu[e]; J = st.R.indices[st.R.indptr[u]:st.R.indptr[u + 1]]; Rl = np.flatnonzero(st.Rel[e])
    la0, lo0 = st.lat[J].mean(), st.lon[J].mean()
    ax.scatter(st.lon[J], st.lat[J], s=14, c='#999999', marker='o', label='Lịch sử', zorder=2)
    ax.scatter(st.lon[Lb[e, :10]], st.lat[Lb[e, :10]], s=16, facecolors='none', edgecolors=C['base'], label='λ = 1', zorder=3)
    ax.scatter(st.lon[Ls[e, :10]], st.lat[Ls[e, :10]], s=16, c=C['rr'], marker='^', label=f'λ* = {lam["main"]}', zorder=4)
    ax.scatter(st.lon[Rl], st.lat[Rl], s=30, c=C['pn'], marker='x', label='Liên quan (Test)', zorder=5)
    ax.plot(lo0, la0, '*', ms=12, color='black', label='Vị trí đại diện', zorder=6)
    ax.set_title(nm, fontsize=8.5); ax.set_aspect(1 / np.cos(np.radians(la0)), adjustable='datalim'); ax.tick_params(labelsize=6)
    from matplotlib.ticker import MaxNLocator; ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(4))
    hb, hs = int(st.Rel[e, Lb[e, :10]].sum()), int(st.Rel[e, Ls[e, :10]].sum())
    crow.append(dict(case=nm, n_hist=int(J.size), mean_rating=float(st.umean[u]), n_rel=int(Rl.size), hits_lambda1=hb, hits_lambdastar=hs,
                     dist_lambda1=float(Dm[e, Lb[e, :10]].mean()), dist_lambdastar=float(Dm[e, Ls[e, :10]].mean()),
                     ref_dist=float(Dm[e, Rl].mean()), sep_km=float(ev.sep[e]), ndcg_lambda1=float(PU[f'{base} λ=1']['N'][e]), ndcg_lambdastar=float(PU[f'RR-100 λ*={lam["main"]}']['N'][e])))
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=5, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(f'{out}/fig311_cases.png'); plt.close(fig)
pd.DataFrame(crow).to_csv(f'{out}/B316_cases.csv', index=False)
json.dump(NUM, open(f'{out}/numbers.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(NUM, indent=1, ensure_ascii=False))
