"""Gom số liệu cho Chương 4 từ mô hình đã đóng gói, API và kết quả TN7. Dùng: python ch4_numbers.py MODEL_DIR TN7_API.json TN7_HTTP.json OUT.json"""
import sys, json, numpy as np
from app import create_app
model_dir, f_api, f_http, out = sys.argv[1:5]
app = create_app(model_dir); c = app.test_client(); rec = app.config['REC']
z = np.load(f'{model_dir}/model.npz')
comp = {k: dict(shape=list(z[k].shape), dtype=str(z[k].dtype), mb=z[k].nbytes / 2**20) for k in z.files}
meta = c.get('/api/meta').get_json(); u = meta['demo_users'][0]
def items(j): return [dict(rank=x['rank'], name=x['name'], dist_km=x.get('dist_km'), s=x.get('s'), g=x.get('g'), score=x.get('score'), has=x.get('has_neighbour')) for x in j['items']]
J = {}
for tag, q in (('star', f'user_id={u}&k=10'), ('lam1', f'user_id={u}&k=10&lam=1'), ('popnear', f'user_id={u}&k=10&mode=popnear'),
               ('new_point', 'lat=39.7684&lon=-86.1581&k=10'), ('new_nopoint', 'k=10'), ('selected', f'user_id={u}&k=10&lat=39.9&lon=-86.0')):
    j = c.get('/api/recommend?' + q).get_json()
    J[tag] = dict(branch=j['branch'], label=j['source_label'], anchor_source=j['anchor_source'], lam=j['lam'], n_history=len(j['history']), anchor=j['anchor'], items=items(j),
                  avg_dist=float(np.mean([x['dist_km'] for x in j['items']])) if j['items'][0].get('dist_km') is not None else None)
nnz_sim = int((rec.sim > 0).sum()); res = dict(meta=meta, demo_user=u, components=comp, examples=J,
           sim_positive_share=nnz_sim / rec.sim.size, dense_rhat_all_users_mb=len(rec.users) * rec.nI * 8 / 2**20,
           api=json.load(open(f_api)), http=json.load(open(f_http)))
json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ('api', 'components', 'examples', 'meta')}, indent=1))
for t, e in J.items(): print(t, e['branch'], e['lam'], e['n_history'], e['avg_dist'], [x['name'][:16] for x in e['items'][:3]])
