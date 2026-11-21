"""TN7 (phần tích hợp, mục 4.7.1 và 4.7.2): so khớp kết quả của API prototype với quy trình ngoại tuyến của Chương 3,
chạy bộ ca kiểm thử chức năng và đo thời gian phản hồi.
Dùng:  python tn7_api_check.py REVIEWS.csv BUSINESS.csv RESULTS_DIR MODEL_DIR OUT.json"""
import os, sys, json, time, argparse, resource
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('DINEREC_CODE', os.path.join(HERE, '..', 'code')))
import run_experiments as X
from app import create_app

rev, biz, res, model_dir, out = sys.argv[1:6]
dec = json.load(open(os.path.join(res, 'decisions_from_val.json')))
k, tau = dec['best']['ItemKNN'][0]; lam, lamP = dec['lam']['main'], dec['lam_P']
a = argparse.Namespace(reviews=rev, business=biz, start='2018-01-01', nmin=5)
rv, bz, tr, va, te, T1, T2 = X.load_split(a)
st = X.Stage(pd.concat([tr, va]), te, bz, 5)

# ---- danh sách ngoại tuyến (đúng các hàm của Chương 3)
Rh, Has = X.m_itemknn(st, k, tau); rk, nc = X.rank_all(st, Rh, Has)
off = {l: X.rerank(st, rk, nc, Rh, l, Has=Has)[:, :10] for l in (1.0, lam, 0.0)}
Rp = X.m_mostpop(st); rkp, ncp = X.rank_all(st, *Rp)
off_pn = X.rerank(st, rkp, ncp, Rp[0], lamP)[:, :10]; off_mp = rkp[:, :10]
items = np.array(st.items); raw = [st.ts.to_raw_uid(int(u)) for u in st.eu]

t0 = time.time(); app = create_app(model_dir); load_s = time.time() - t0
c = app.test_client(); rec = app.config['REC']
res_ = {}; lat_ms = []
def check(name, offline, q):
    ok = 0
    for e, u in enumerate(raw):
        t = time.perf_counter(); j = c.get(f'/api/recommend?user_id={u}&k=10&' + q).get_json(); lat_ms.append((time.perf_counter() - t) * 1000)
        ok += [x['business_id'] for x in j['items']] == list(items[offline[e]])
    res_[name] = dict(users=len(raw), identical_lists=ok, share=ok / len(raw))
for l in (1.0, lam, 0.0): check(f'ItemKNN+rerank lam={l}', off[l], f'lam={l}')
check(f'PopNear lam_P={lamP}', off_pn, 'mode=popnear')
check('MostPopular', off_mp, 'mode=mostpop')
# điểm dự đoán: API so với ma trận ngoại tuyến
dmax = max(float(np.abs(rec.predict(rec.uid[u])[0] - Rh[e]).max()) for e, u in enumerate(raw))
hmis = sum(int((rec.predict(rec.uid[u])[1] != Has[e]).sum()) for e, u in enumerate(raw))

# ---- ca kiểm thử chức năng
u0 = raw[0]; nh0 = int(st.R.getnnz(axis=1)[st.eu[0]])
def j_(q): r = c.get('/api/recommend?' + q); return r.status_code, r.get_json()
cases = []
def case(cid, desc, q, cond):
    code, j = j_(q); okk = bool(cond(code, j)); cases.append(dict(id=cid, desc=desc, query=q, status=code, passed=okk))
case('TC1', 'Người dùng đã biết, mặc định', f'user_id={u0}', lambda s, j: s == 200 and j['branch'] == 'itemknn_rerank' and j['lam'] == lam and len(j['items']) == 10 and len(j['history']) == nh0)
case('TC2', 'Không khuyến nghị nhà hàng đã đánh giá', f'user_id={u0}&k=50', lambda s, j: s == 200 and not ({x['business_id'] for x in j['items']} & {h['business_id'] for h in j['history']}))
case('TC3', 'λ = 1 cho đúng thứ tự của mô hình nền', f'user_id={u0}&lam=1', lambda s, j: [x['business_id'] for x in j['items']] == list(items[rk[0, :10]]))
case('TC4', 'Khoảng cách trung bình không tăng khi λ giảm từ 1 về 0', f'user_id={u0}', lambda s, j: np.mean([x['dist_km'] for x in j_(f'user_id={u0}&lam=0')[1]['items']]) <= np.mean([x['dist_km'] for x in j_(f'user_id={u0}&lam=1')[1]['items']]) + 1e-9)
case('TC5', 'Người dùng mới có vị trí: PopNear quanh vị trí chọn', 'lat=39.7684&lon=-86.1581', lambda s, j: s == 200 and j['branch'] == 'popnear_new_user' and j['lam'] == lamP and len(j['items']) == 10)
case('TC6', 'Người dùng mới không có vị trí: MostPopular', '', lambda s, j: s == 200 and j['branch'] == 'mostpop_new_user' and [x['business_id'] for x in j['items']] == list(rec.items[rec.pop_rank[:10]]))
case('TC7', 'user_id không có trong dữ liệu được xử lý như người dùng mới và được đánh dấu', 'user_id=khong_ton_tai', lambda s, j: s == 200 and j['unknown_user_id'] and j['branch'] == 'mostpop_new_user')
case('TC8', 'K = 20 trả về 20 nhà hàng khác nhau', f'user_id={u0}&k=20', lambda s, j: s == 200 and len({x['business_id'] for x in j['items']}) == 20)
case('TC9', 'λ ngoài [0, 1] bị từ chối', f'user_id={u0}&lam=1.5', lambda s, j: s == 400)
case('TC10', 'K ngoài [1, 50] bị từ chối', f'user_id={u0}&k=0', lambda s, j: s == 400)
case('TC11', 'Thiếu một trong hai tọa độ bị từ chối', 'lat=39.7', lambda s, j: s == 400)
case('TC12', 'Tọa độ không hợp lệ bị từ chối', 'lat=120&lon=0', lambda s, j: s == 400)
case('TC13', 'Tham số không phải số bị từ chối', f'user_id={u0}&lam=abc', lambda s, j: s == 400)
nhE = st.R.getnnz(axis=1)[st.eu]; e4 = int(np.flatnonzero((nhE >= 1) & (nhE <= 4))[0]); u4 = raw[e4]
case('TC15', 'Người dùng có 1–4 tương tác nhận danh sách cá nhân hóa, không chứa nhà hàng đã đánh giá', f'user_id={u4}',
     lambda s, j: s == 200 and j['branch'] == 'itemknn_rerank' and 1 <= len(j['history']) <= 4 and len(j['items']) == 10
     and [x['business_id'] for x in j['items']] == list(items[off[lam][e4]]) and not ({x['business_id'] for x in j['items']} & {h['business_id'] for h in j['history']}))
case('TC16', 'Người dùng đã biết chọn vị trí khác: vị trí đó thay vị trí đại diện, nhãn vẫn là cá nhân hóa', f'user_id={u0}&lat=39.9&lon=-86.0',
     lambda s, j: s == 200 and j['branch'] == 'itemknn_rerank' and j['anchor_source'] == 'selected' and abs(j['anchor']['lat'] - 39.9) < 1e-12 and abs(j['anchor']['lon'] + 86.0) < 1e-12 and j['source_label'] == 'cá nhân hóa')
case('TC17', 'Nhãn nguồn đúng ở ba nhánh', f'user_id={u0}',
     lambda s, j: j['source_label'] == 'cá nhân hóa' and j_('lat=39.77&lon=-86.16')[1]['source_label'] == 'phổ biến gần bạn' and j_('')[1]['source_label'] == 'phổ biến')
case('TC18', 'λ = 0: danh sách xếp theo khoảng cách tăng dần', f'user_id={u0}&lam=0',
     lambda s, j: s == 200 and all(a_['dist_km'] <= b_['dist_km'] + 1e-12 for a_, b_ in zip(j['items'], j['items'][1:])))
case('TC14', 'Cùng yêu cầu cho cùng kết quả (tất định)', f'user_id={u0}&lam=0.5', lambda s, j: j == j_(f'user_id={u0}&lam=0.5')[1] or [x['business_id'] for x in j['items']] == [x['business_id'] for x in j_(f'user_id={u0}&lam=0.5')[1]['items']])

L = np.array(lat_ms)
outj = dict(compare=res_, rhat_max_abs_diff=dmax, has_mismatch=hmis, cases=cases, n_cases=len(cases), n_passed=sum(x['passed'] for x in cases),
            latency_ms=dict(n=int(L.size), p50=float(np.percentile(L, 50)), p95=float(np.percentile(L, 95)), max=float(L.max())),
            load_seconds=load_s, model_file_mb=os.path.getsize(os.path.join(model_dir, 'model.npz')) / 2**20,
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
json.dump(outj, open(out, 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k_: v for k_, v in outj.items() if k_ != 'cases'}, indent=1, ensure_ascii=False))
for x in cases: print(x['id'], 'ĐẠT' if x['passed'] else 'KHÔNG ĐẠT', x['desc'])
