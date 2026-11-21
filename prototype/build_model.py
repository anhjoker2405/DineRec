"""Đóng gói mô hình cho prototype DineRec (mục 4.3).
Huấn luyện lại ItemKNN trên H_test = Train ∪ Val với đúng cấu hình đã khóa ở Chương 3 (decisions_from_val.json),
rồi lưu những gì cần để phục vụ trực tuyến: ma trận tương đồng nhà hàng, điểm trung bình nhà hàng, ma trận đánh giá,
danh mục nhà hàng (tên, tọa độ, pop+), thứ tự phá hòa và các tham số lambda.

Dùng:  python build_model.py REVIEWS.csv BUSINESS.csv RESULTS_DIR OUT_DIR
       (cần run_experiments.py và dinerec_core.py ở thư mục code, xem CODE_DIR bên dưới)
"""
import sys, os, json, time, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.environ.get('DINEREC_CODE', os.path.join(HERE, '..', 'code'))
sys.path.insert(0, CODE_DIR)
import run_experiments as X                                    # dùng lại đúng hàm chia dữ liệu và lớp Stage của thực nghiệm
from surprise import KNNWithMeans


def main():
    reviews, business, results, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    dec = json.load(open(os.path.join(results, 'decisions_from_val.json')))
    k, tau = dec['best']['ItemKNN'][0]
    lam, lamP = dec['lam']['main'], dec['lam_P']
    assert dec['base'] == 'ItemKNN', 'prototype giả định mô hình nền là ItemKNN'
    a = argparse.Namespace(reviews=reviews, business=business, start='2018-01-01', nmin=5)
    rv, bz, tr, va, te, T1, T2 = X.load_split(a)
    st = X.Stage(pd.concat([tr, va]), te, bz, 5)                 # giai đoạn báo cáo: H_test, danh mục I_H
    al = KNNWithMeans(k=k, min_k=1, sim_options={'name': 'pearson', 'user_based': False, 'min_support': tau}, verbose=False)
    al.fit(st.ts)
    names = np.array([str(x) for x in bz.set_index('business_id').loc[st.items, 'name'].fillna('')], dtype='U')
    users = np.array([st.ts.to_raw_uid(i) for i in range(st.nU)], dtype=str)
    np.savez_compressed(os.path.join(out, 'model.npz'),
                        sim=np.asarray(al.sim), item_mean=np.asarray(al.means),
                        R_data=st.R.data, R_indices=st.R.indices, R_indptr=st.R.indptr,
                        items=np.array(st.items, dtype=str), names=names, lat=st.lat, lon=st.lon,
                        pop_pos=st.pop_pos, order=st.order, users=users)
    meta = dict(base='ItemKNN', k=int(k), tau=int(tau), min_k=1, M=100, K=10, lam=float(lam), lam_P=float(lamP),
                T1=str(T1.date()), T2=str(T2.date()), n_users=int(st.nU), n_items=int(st.nI), n_ratings=int(st.R.nnz),
                build_seconds=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
