"""Hình 3 của bài báo (khổ một cột): đường đánh đổi NDCG@10 theo AvgDist@10 khi λ chạy trên lưới.
Cùng dữ liệu và cách vẽ với Hình 3.9 của khóa luận; chỉ đổi tên trục theo thuật ngữ của bài (AvgDist@10)
và đặt lại nhãn λ để không chồng nhau ở khổ 3,5 inch.
Chạy: python paper_fig3.py <thư mục results3> <thư mục code có vn_format.py> <tệp png ra>"""
import sys, json
res, code, outp = sys.argv[1:4]
sys.path.insert(0, code)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import vn_format  # dấu phẩy thập phân cho mọi nhãn số

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
C = dict(base='#1f5a96', rr='#d9822b')
dec = json.load(open(f'{res}/decisions_from_val.json')); lam = dec['lam']['main']
lv = pd.read_csv(f'{res}/TN3_lambda_val.csv'); lv = lv[lv.variant == 'main']
tt = pd.read_csv(f'{res}/TN2_TN3_TN4_test.csv'); tt = tt[tt.model.str.match(r'^\w+\+rerank λ=\d\.\d$')].copy()
tt['lambda'] = tt.model.str.extract(r'λ=(\d\.\d)').astype(float)

fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.plot(lv['AvgDist@10'], lv['NDCG@10'], 'o-', color=C['base'], ms=3.5, label='Val (chọn λ)')
ax.plot(tt['AvgDist@10'], tt['NDCG@10'], 's--', color=C['rr'], ms=3, label='Test (chỉ báo cáo)')
k_ = lv['lambda'].sub(lam).abs().idxmin()
ax.plot(lv.loc[k_, 'AvgDist@10'], lv.loc[k_, 'NDCG@10'], '*', ms=12, color=C['rr'], mec='black', label=f'λ* = {lam}')
offs = {0.0: (-4, -11), 0.1: (5, -3), 0.2: (6, -1), 0.5: (0, 6), 0.9: (4, 3), 1.0: (-4, 5)}  # chỉ ghi nhãn các λ này
for _, r in lv.iterrows():
    l_ = round(r['lambda'], 1)
    if l_ in offs:
        ax.annotate(f'{l_:.1f}', (r['AvgDist@10'], r['NDCG@10']), fontsize=7, xytext=offs[l_], textcoords='offset points',
                    color='#c0392b' if not r.ok else 'black', ha='center' if offs[l_][0] < 0 else 'left')
ax.set_xlabel('AvgDist@10 (km)'); ax.set_ylabel('NDCG@10')
ax.legend(frameon=False, fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(outp)
print('λ*', lam, 'Val pts', len(lv), 'Test pts', len(tt))
print(lv[['lambda', 'AvgDist@10', 'NDCG@10', 'ok']].round(4).to_string(index=False))
