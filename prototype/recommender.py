"""Lõi phục vụ khuyến nghị của prototype DineRec (Thuật toán 4.1).

Ba nhánh:
  1. người dùng đã có lịch sử trong H  -> ItemKNN (điểm chưa cắt) -> khóa xếp hạng hai tầng -> M ứng viên -> xếp hạng lại theo khoảng cách
  2. người dùng mới, có vị trí chọn trên bản đồ -> PopNear quanh vị trí đó
  3. người dùng mới, không có vị trí            -> MostPopular
Phép dự đoán ItemKNN chép đúng hàm m_itemknn của run_experiments.py; hàm xếp hạng lại là geo_rerank của dinerec_core.py
(dùng chung với thực nghiệm). Tính đúng được kiểm bằng tn7_api_check.py.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get('DINEREC_CODE', os.path.join(HERE, '..', 'code')))
from dinerec_core import geo_rerank, haversine


class DineRec:
    def __init__(self, model_dir):
        t0 = time.time()
        z = np.load(os.path.join(model_dir, 'model.npz'), allow_pickle=False)
        self.meta = json.load(open(os.path.join(model_dir, 'meta.json')))
        self.sim, self.mean = z['sim'], z['item_mean']
        self.Rd, self.Ri, self.Rp = z['R_data'], z['R_indices'], z['R_indptr']
        self.items, self.names = z['items'], z['names']
        self.lat, self.lon = z['lat'], z['lon']
        self.pop, self.order = z['pop_pos'], z['order']
        self.users = z['users']
        self.uid = {u: i for i, u in enumerate(self.users)}
        self.nI = len(self.items)
        self.k, self.min_k, self.M = self.meta['k'], self.meta['min_k'], self.meta['M']
        self.lam, self.lam_P, self.K = self.meta['lam'], self.meta['lam_P'], self.meta['K']
        # thứ tự MostPopular cho người dùng mới: pop+ giảm dần, rồi business_id tăng dần
        self.pop_rank = np.lexsort((self.order, -np.round(self.pop, 9)))
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------------ dữ liệu của một người dùng
    def history(self, u):
        a, b = self.Rp[u], self.Rp[u + 1]
        return self.Ri[a:b], self.Rd[a:b]

    def anchor(self, J):
        """Vị trí đại diện: trung bình cộng tọa độ các nhà hàng đã đánh giá (mục 2.5.1)."""
        return float(self.lat[J].mean()), float(self.lon[J].mean())

    # ------------------------------------------------------------------ ItemKNN cho một người dùng (giống m_itemknn)
    def predict(self, u):
        J, rj = self.history(u)
        S = np.where(self.sim[:, J] > 0, self.sim[:, J], 0.0)
        if J.size > self.k:
            idx = np.argpartition(-S, self.k - 1, axis=1)[:, :self.k]
            keep = np.zeros_like(S, bool); np.put_along_axis(keep, idx, True, 1); S = S * keep
        cnt = (S > 0).sum(1); den = S.sum(1); num = (S * (rj - self.mean[J])[None, :]).sum(1)
        rhat = self.mean.copy(); ok = cnt >= self.min_k
        rhat[ok] += num[ok] / den[ok]
        return rhat, cnt >= 1

    # ------------------------------------------------------------------ xếp hạng ứng viên và xếp hạng lại
    def _rank(self, score, has, exclude):
        """Khóa (2.8) trên các nhà hàng chưa đánh giá: có láng giềng, điểm (làm tròn 9 chữ số) giảm dần, business_id tăng dần."""
        seen = np.zeros(self.nI, bool); seen[exclude] = True
        cand = np.flatnonzero(~seen)
        o = np.lexsort((self.order[cand], -np.round(score[cand], 9), -has[cand].astype(int)))
        return cand[o]

    def _rerank(self, ranked, score, has, la0, lo0, K, lam):
        c = ranked[:min(self.M, ranked.size)]
        d = haversine(la0, lo0, self.lat, self.lon)[c]
        sel, s, g = geo_rerank(score[c], d, K, lam, 'minmax', None, self.order[c], has[c])
        comb = lam * s + (1 - lam) * g
        return [dict(item=int(c[j]), dist_km=float(d[j]), score_model=float(score[c[j]]), has_neighbour=bool(has[c[j]]),
                     s=float(s[j]), g=float(g[j]), score=float(comb[j])) for j in sel]

    def _out(self, rows):
        for r, x in enumerate(rows, 1):
            i = x.pop('item')
            x.update(rank=r, business_id=str(self.items[i]), name=str(self.names[i]), lat=float(self.lat[i]), lon=float(self.lon[i]))
        return rows

    # ------------------------------------------------------------------ Thuật toán 4.1
    LABEL = {'itemknn_rerank': 'cá nhân hóa', 'popnear': 'phổ biến gần bạn', 'popnear_new_user': 'phổ biến gần bạn',
             'mostpop': 'phổ biến', 'mostpop_new_user': 'phổ biến'}

    def recommend(self, user_id=None, lat=None, lon=None, K=None, lam=None, mode='auto'):
        """mode: auto | popnear | mostpop. 'popnear' và 'mostpop' cho người dùng đã biết dùng để so sánh trên giao diện."""
        K = self.K if K is None else int(K)
        t0 = time.perf_counter()
        u = self.uid.get(user_id) if user_id is not None else None
        if u is not None:
            J, rj = self.history(u)
            # vị trí đại diện: mặc định suy từ lịch sử; nếu người dùng chọn vị trí trên bản đồ thì dùng vị trí đó
            # (khi đó kết quả khác với đánh giá ngoại tuyến, vốn luôn dùng a_u)
            la0, lo0 = (float(lat), float(lon)) if lat is not None and lon is not None else self.anchor(J)
            anchor_src = 'selected' if lat is not None and lon is not None else 'history'
            if mode == 'auto':
                lam = self.lam if lam is None else float(lam)
                rhat, has = self.predict(u)
                rows = self._rerank(self._rank(rhat, has, J), rhat, has, la0, lo0, K, lam)
                branch = 'itemknn_rerank'
            elif mode == 'popnear':
                lam = self.lam_P if lam is None else float(lam)
                one = np.ones(self.nI, bool)
                rows = self._rerank(self._rank(self.pop, one, J), self.pop, one, la0, lo0, K, lam)
                branch = 'popnear'
            else:
                one = np.ones(self.nI, bool); r = self._rank(self.pop, one, J)[:K]
                d = haversine(la0, lo0, self.lat, self.lon)
                rows = [dict(item=int(i), dist_km=float(d[i]), score_model=float(self.pop[i])) for i in r]
                branch, lam = 'mostpop', None
            hist = [dict(business_id=str(self.items[j]), name=str(self.names[j]), lat=float(self.lat[j]), lon=float(self.lon[j]),
                         stars=float(s)) for j, s in zip(J, rj)]
            anchor = dict(lat=la0, lon=lo0)
        elif lat is not None and lon is not None:
            la0, lo0 = float(lat), float(lon)
            lam = self.lam_P if lam is None else float(lam)
            one = np.ones(self.nI, bool)
            rows = self._rerank(self.pop_rank, self.pop, one, la0, lo0, K, lam)
            branch, hist, anchor, anchor_src = 'popnear_new_user', [], dict(lat=la0, lon=lo0), 'selected'
        else:
            rows = [dict(item=int(i), score_model=float(self.pop[i])) for i in self.pop_rank[:K]]
            branch, hist, anchor, lam, anchor_src = 'mostpop_new_user', [], None, None, None
        return dict(user_id=user_id, known_user=u is not None, unknown_user_id=user_id is not None and u is None, branch=branch,
                    source_label=self.LABEL[branch], K=K, lam=lam, anchor=anchor, anchor_source=anchor_src,
                    history=hist, items=self._out(rows), ms=round((time.perf_counter() - t0) * 1000, 2))
