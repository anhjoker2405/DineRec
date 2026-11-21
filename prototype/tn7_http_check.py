"""TN7 (đo qua HTTP, mục 4.7.3): gửi 300 yêu cầu cho mỗi nhánh tới máy chủ đang chạy (python app.py model PORT).
Dùng:  python tn7_http_check.py http://127.0.0.1:5000 MODEL_DIR OUT.json [PID_MÁY_CHỦ]
PID chỉ dùng trên Linux để đọc bộ nhớ đỉnh của máy chủ từ /proc/PID/status."""
import sys, json, time, random, urllib.request
import numpy as np

base, model_dir, out = sys.argv[1:4]
pid = sys.argv[4] if len(sys.argv) > 4 else None
users = list(np.load(f'{model_dir}/model.npz')['users'])
rng = random.Random(0); us = rng.sample(users, 300)


def t(q):
    s = time.perf_counter(); urllib.request.urlopen(base + '/api/recommend?' + q).read(); return (time.perf_counter() - s) * 1000


res = {}
for name, qs in (('known_user', [f'user_id={u}' for u in us]),
                 ('new_user_point', [f'lat={39.6 + 0.4 * rng.random():.4f}&lon={-86.35 + 0.4 * rng.random():.4f}' for _ in range(300)]),
                 ('new_user_no_point', [''] * 300)):
    L = np.array([t(q) for q in qs]); res[name] = dict(n=len(L), p50=float(np.percentile(L, 50)), p95=float(np.percentile(L, 95)), max=float(L.max()))
if pid:
    st = open(f'/proc/{pid}/status').read().splitlines()
    g = lambda k: int([l for l in st if l.startswith(k)][0].split()[1]) / 1024
    res['server_peak_rss_mb'], res['server_rss_mb'] = g('VmHWM'), g('VmRSS')
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
