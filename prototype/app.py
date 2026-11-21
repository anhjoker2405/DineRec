"""Máy chủ web của prototype DineRec (mục 4.5).
Chạy:  python app.py MODEL_DIR [PORT]      rồi mở http://127.0.0.1:5000
"""
import os, sys
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from recommender import DineRec

HERE = os.path.dirname(os.path.abspath(__file__))


def create_app(model_dir):
    rec = DineRec(model_dir)
    app = Flask(__name__, static_folder=os.path.join(HERE, 'static'))
    app.config['REC'] = rec
    nh = np.diff(rec.Rp)
    rng = np.random.default_rng(0)
    demo = [str(rec.users[i]) for i in sorted(rng.choice(np.flatnonzero(nh >= 5), 12, replace=False))]

    def bad(msg):
        return jsonify(error=msg), 400

    @app.get('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    @app.get('/api/meta')
    def meta():
        return jsonify(rec.meta | dict(center=dict(lat=float(np.median(rec.lat)), lon=float(np.median(rec.lon))), demo_users=demo))

    @app.get('/api/user/<user_id>')
    def user(user_id):
        u = rec.uid.get(user_id)
        if u is None:
            return jsonify(error='người dùng không có lịch sử trong dữ liệu huấn luyện'), 404
        return jsonify(user_id=user_id, n_history=int(nh[u]))

    @app.get('/api/recommend')
    def recommend():
        q = request.args
        try:
            K = int(q.get('k', rec.K))
            lam = q.get('lam'); lam = None if lam in (None, '') else float(lam)
            lat, lon = q.get('lat'), q.get('lon')
            lat = None if lat in (None, '') else float(lat); lon = None if lon in (None, '') else float(lon)
        except ValueError:
            return bad('tham số không hợp lệ')
        mode = q.get('mode', 'auto')
        if not 1 <= K <= 50: return bad('k phải trong [1, 50]')
        if lam is not None and not 0.0 <= lam <= 1.0: return bad('lam phải trong [0, 1]')
        if (lat is None) != (lon is None): return bad('cần cả lat và lon')
        if lat is not None and not (-90 <= lat <= 90 and -180 <= lon <= 180): return bad('tọa độ không hợp lệ')
        if mode not in ('auto', 'popnear', 'mostpop'): return bad('mode phải là auto, popnear hoặc mostpop')
        uid = q.get('user_id') or None
        return jsonify(rec.recommend(user_id=uid, lat=lat, lon=lon, K=K, lam=lam, mode=mode))

    return app


if __name__ == '__main__':
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'model')
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    create_app(model_dir).run(host='127.0.0.1', port=port, debug=False)
