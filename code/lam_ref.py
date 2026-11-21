"""Phân tích độ nhạy ở mục 3.7: chọn λ có hiệu chỉnh theo khoảng cách tham chiếu, công thức (3.1).
    λ** = argmin_{λ ∈ Λδ} |d̃_λ − d̃_ref|,  Λδ = các λ đạt tiêu chí (2.18) trên Val.
d̃_λ: trung vị khoảng cách (gộp mọi cặp người dùng–nhà hàng) từ vị trí đại diện đến các nhà hàng trong top-10 ứng với λ.
d̃_ref: trung vị khoảng cách (gộp) đến các nhà hàng liên quan R_u của cùng tập đo (cùng cách tính 7,0 km ở ch3_analysis.py).
Hai lưới: lưới đã khóa của Thuật toán 2.6 và lưới làm mịn trong [0,9; 1] bước 0,01 (thăm dò). Tiêu chí không thua kém dùng
cùng mẫu bootstrap với Thuật toán 2.6 (hạt giống 0, B = 1.000). Chỉ chọn trên Val; Test chỉ để báo cáo. Không huấn luyện lại
gì ngoài ItemKNN với cấu hình đã khóa. Script cũng tách nguyên nhân bước nhảy khoảng cách giữa λ = 1 và λ = 0,99.
Dùng (trong thư mục code): python lam_ref.py indy_reviews.csv indy_business.csv results3 results3/lam_ref.json"""
import sys, json, argparse
import numpy as np, pandas as pd
import run_experiments as X

rev, biz, res, outp = sys.argv[1:5]
dec = json.load(open(f'{res}/decisions_from_val.json'))
assert dec['base'] == 'ItemKNN'
cfg = tuple(dec['best']['ItemKNN'][0]); lam_star = dec['lam']['main']
a = argparse.Namespace(reviews=rev, business=biz, start='2018-01-01', nmin=5)
rv, bz, tr, va, te, T1, T2 = X.load_split(a)
COARSE = X.LAMS                                                            # 1; 0,9; ...; 0
FINE = [1.0] + [round(1 - k / 100, 2) for k in range(1, 11)]                # 1; 0,99; ...; 0,9
GRIDS = {'coarse': COARSE, 'fine': FINE}


def run_stage(s):
    Rh, Has = X.m_itemknn(s, *cfg); rk, nc = X.rank_all(s, Rh, Has); Dm = s.dist('mean')
    lam_all = sorted(set(COARSE) | set(FINE), reverse=True)
    L = {l: X.rerank(s, rk, nc, Rh, l, Has=Has)[:, :10] for l in lam_all}
    N = {l: X.per_user(s, L[l], 10)[2] for l in lam_all}
    ref = Dm[s.Rel]
    return dict(Rh=Rh, Has=Has, Dm=Dm, L=L, N=N, ref_median=float(np.median(ref)), ref_le5=float((ref <= 5).mean()))


def grid_table(s, R, grid):
    """Bảng theo λ của một lưới; Q95 theo (2.18) so với λ = 1, cùng chỉ số bootstrap với Thuật toán 2.6."""
    nd = np.array([R['N'][l] for l in grid]); idx = np.random.default_rng(0).integers(0, s.nE, (1000, s.nE))
    boot = nd[:, idx].mean(2); q = np.quantile((boot[0] - boot) / boot[0], 0.95, axis=1); rows = []
    for k, l in enumerate(grid):
        d = np.take_along_axis(R['Dm'], R['L'][l], 1)
        rows.append(dict(lam=l, ndcg=float(R['N'][l].mean()), median=float(np.median(d)), mean=float(d.mean()),
                         le5=float((d <= 5).mean()), Q95=float(q[k]), ok=bool(q[k] < 0.05)))
    T = pd.DataFrame(rows); T['gap'] = (T['median'] - R['ref_median']).abs()
    return T


def jump(s, R):
    """Người dùng có nhà hàng không có láng giềng hợp lệ trong danh sách λ = 1, và trung vị khoảng cách ở λ = 1 và 0,99."""
    full = np.take_along_axis(R['Has'], R['L'][1.0], 1).all(1); out = dict(n_users=int(s.nE), n_fallback=int((~full).sum()))
    for nm, m in (('fallback', ~full), ('full', full)):
        out[nm] = {str(l): float(np.median(np.take_along_axis(R['Dm'], R['L'][l], 1)[m])) for l in (1.0, 0.99)}
    return out


OUT = dict(config=dict(itemknn=cfg, lam_star=lam_star))
sv = X.Stage(tr, va, bz, 5); Rv = run_stage(sv)
saved = pd.read_csv(f'{res}/TN3_lambda_val.csv'); saved = saved[saved.variant == 'main'].set_index('lambda')
OUT['val'] = dict(ref_median=Rv['ref_median'], ref_le5=Rv['ref_le5'], jump=jump(sv, Rv))
for g, grid in GRIDS.items():
    T = grid_table(sv, Rv, grid)
    if g == 'coarse':                                                       # phải trùng bảng đã lưu của Thuật toán 2.6
        assert (T.ok.values == saved.loc[T.lam, 'ok'].values).all()
        OUT['val']['coarse_maxdiff_vs_saved'] = float(np.abs(T[['ndcg', 'mean', 'Q95']].values - saved.loc[T.lam, ['NDCG@10', 'AvgDist@10', 'Q95_loss']].values).max())
    ok = T[T.ok].sort_values(['gap', 'lam'], ascending=[True, False])     # hòa: chọn λ lớn hơn
    OUT['val'][g] = dict(lam2=float(ok.lam.iloc[0]), rows=T.to_dict('records'))
lam2 = OUT['val']['coarse']['lam2']; assert lam2 == OUT['val']['fine']['lam2']

st = X.Stage(pd.concat([tr, va]), te, bz, 5); Rt = run_stage(st)
cmp = lambda a_, b_: {k: float(v) for k, v in X.compare(a_, b_, 1000).items()}
d2 = np.take_along_axis(Rt['Dm'], Rt['L'][lam2], 1)
OUT['test'] = dict(ref_median=Rt['ref_median'], ref_le5=Rt['ref_le5'], jump=jump(st, Rt),
                   lam2=dict(lam=lam2, ndcg=float(Rt['N'][lam2].mean()), median=float(np.median(d2)), mean=float(d2.mean()), le5=float((d2 <= 5).mean())),
                   lam2_vs_l1=cmp(Rt['N'][lam2], Rt['N'][1.0]) if lam2 != 1.0 else dict(diff=0.0, note='λ** = 1: trùng mô hình nền'), lam2_vs_lamstar=cmp(Rt['N'][lam2], Rt['N'][lam_star]),
                   fine_rows=grid_table(st, Rt, FINE).to_dict('records'))
json.dump(OUT, open(outp, 'w'), indent=1, ensure_ascii=False)
for g in GRIDS: print(g, 'λ** =', OUT['val'][g]['lam2']); print(pd.DataFrame(OUT['val'][g]['rows']).round(4).to_string(index=False))
print('Val d_ref', round(Rv['ref_median'], 3), 'jump', OUT['val']['jump']); print('Test', {k: v for k, v in OUT['test'].items() if k != 'fine_rows'})
