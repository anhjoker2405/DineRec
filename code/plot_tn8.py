"""Bảng 3.19 và Hình 3.10 từ tệp tn8.jsonl do tn8_real.py ghi ra.
Dùng: python plot_tn8.py TN8_JSONL OUT_DIR"""
import sys, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import vn_format  # dấu phẩy thập phân cho mọi hình (quy ước tiếng Việt)
from matplotlib.ticker import FixedLocator, NullLocator
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
rows = [json.loads(l) for l in open(src) if l.startswith('{')]
for r in rows: r['delta_GB'] = r['peak_GB'] - r['peak_before_GB']
pd.DataFrame(rows).to_csv(f'{out}/B319_tn8.csv', index=False)
S = sorted([r for r in rows if r['mode'] == 'surprise'], key=lambda r: r['n_users']); O = [r for r in rows if r['mode'] == 'ours']
per = S[-1]['delta_GB'] * 1e9 / S[-1]['n_users'] ** 2
fig, ax = plt.subplots(figsize=(5.4, 3.1))
ax.loglog([r['n_users'] for r in S], [r['delta_GB'] for r in S], 'o', color='#1f5a96', label='Surprise KNNWithMeans (đo)')
xs = np.logspace(np.log10(1500), np.log10(40000), 50); ax.loglog(xs, 55 * xs ** 2 / 1e9, '--', color='#1f5a96', lw=1, label='55·|U|² byte')
nfull = max(r['n_users'] for r in O)
ax.loglog([nfull], [per * nfull ** 2 / 1e9], 'o', mfc='none', color='#1f5a96', label=f'Surprise ngoại suy, |U| = {nfull:,}'.replace(',', '.'))
ok = [r for r in O if r['delta_GB'] > 0.005]; no = [r for r in O if r not in ok]
ax.loglog([r['n_users'] for r in ok], [r['delta_GB'] for r in ok], 's', color='#d9822b', label='Thuật toán 2.2 (đo)')
ax.loglog([r['n_users'] for r in no], [0.01] * len(no), 'v', color='#d9822b', mfc='none', label='Thuật toán 2.2: dưới 0,01 GB')
ax.axhline(32, color='#999999', lw=.8, ls=':'); ax.text(1600, 38, '32 GB RAM', fontsize=7, color='#666666')
t = sorted({r['n_users'] for r in rows}); ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels([f'{x:,}'.replace(',', '.') for x in t])
ax.set_xlabel('Số người dùng |U| (trục log)'); ax.set_ylabel('Bộ nhớ đỉnh tăng thêm (GB, log)'); ax.legend(frameon=False, fontsize=7, loc='center right')
fig.tight_layout(); fig.savefig(f'{out}/fig310_tn8.png')
print(pd.DataFrame(rows)[['mode', 'n_users', 'sec', 'delta_GB', 'max_dsim', 'n_mismatch', 'n_pairs']].to_string(index=False))
print(f'Surprise: {per:.1f} byte/cặp người dùng; ngoại suy {per * nfull ** 2 / 1e9:.0f} GB cho {nfull} người dùng')
