"""Bước 1 (chạy trên máy, thư mục có file JSON của Yelp): lọc nhà hàng Indianapolis, IN -> 2 file CSV nhỏ để upload lên Colab.
Dùng:  python prepare_data.py .
Tạo: indy_reviews.csv, indy_business.csv. In thêm thống kê cửa sổ 2018/2019/2020 để chọn mốc bắt đầu."""
import json, os, sys
import pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else '.'
CITY = 'Indianapolis, IN'; NMIN = 5
CITY_N = CITY.split(',')[0].strip().casefold(); STATE_N = CITY.split(',')[1].strip().upper()  # chuẩn hóa chuỗi: bỏ khoảng trắng thừa, không phân biệt hoa thường
biz = {}; st = dict(business=0, restaurants=0, city=0, coords=0); variants = {}
with open(os.path.join(d, 'yelp_academic_dataset_business.json'), encoding='utf8') as f:
    for line in f:
        b = json.loads(line); st['business'] += 1
        if not (b.get('categories') and 'Restaurants' in b['categories']): continue
        st['restaurants'] += 1
        if not ((b.get('city') or '').strip().casefold() == CITY_N and (b.get('state') or '').strip().upper() == STATE_N): continue
        st['city'] += 1; variants[b['city']] = variants.get(b['city'], 0) + 1
        if not (b.get('latitude') is not None and b.get('longitude') is not None and -90 <= b['latitude'] <= 90 and -180 <= b['longitude'] <= 180): continue
        st['coords'] += 1
        biz[b['business_id']] = (b['name'], b['latitude'], b['longitude'], b['categories'])
rows = []; nrev = 0; nrev_biz = 0; last = ''
with open(os.path.join(d, 'yelp_academic_dataset_review.json'), encoding='utf8') as f:
    for line in f:
        r = json.loads(line); nrev += 1; last = max(last, r['date'])
        if r['business_id'] in biz:
            nrev_biz += 1
            if 1 <= r['stars'] <= 5:
                rows.append((r['review_id'], r['user_id'], r['business_id'], int(r['stars']), r['date']))
df = pd.DataFrame(rows, columns=['review_id', 'user_id', 'business_id', 'stars', 'date']).drop_duplicates('review_id')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date', 'review_id']).drop_duplicates(['user_id', 'business_id'], keep='first')
bz = pd.DataFrame([(k,) + v for k, v in biz.items()], columns=['business_id', 'name', 'latitude', 'longitude', 'categories'])
bz = bz[bz.business_id.isin(set(df.business_id))]
print('=== BẢNG 3.2: số bản ghi sau từng bước ===')
print(f"doanh nghiệp trong Yelp: {st['business']}; có nhãn Restaurants: {st['restaurants']}; thuộc Indianapolis, IN (sau chuẩn hóa): {st['city']}; có tọa độ hợp lệ: {st['coords']}")
print('các cách viết city được gộp:', variants)
print(f"đánh giá trong Yelp: {nrev} (ngày mới nhất {last}); của các nhà hàng trên: {nrev_biz}; số sao hợp lệ: {len(rows)}; "
      f"sau bỏ trùng review_id và cặp (user, nhà hàng): {len(df)}; từ 2018-01-01: {(df.date >= '2018-01-01').sum()}")
print(f"nhà hàng có ít nhất một đánh giá: {df.business_id.nunique()}; người dùng: {df.user_id.nunique()}")
df.to_csv('indy_reviews.csv', index=False); bz.to_csv('indy_business.csv', index=False)
print('Đã ghi indy_reviews.csv:', len(df), 'dòng; indy_business.csv:', len(bz), 'nhà hàng')

def cohort(H, E):
    cnt = H.groupby('business_id').user_id.nunique(); keep = cnt[cnt >= NMIN].index
    H = H[H.business_id.isin(keep)]
    E = E[E.user_id.isin(set(H.user_id)) & E.business_id.isin(keep) & (E.stars >= 4)]
    seen = set(zip(H.user_id, H.business_id))
    E = E[[(u, b) not in seen for u, b in zip(E.user_id, E.business_id)]]
    return E.user_id.nunique(), len(E), H
for st in ['2018-01-01', '2019-01-01', '2020-01-01']:
    c = df[df.date >= st]; ds = c.date.sort_values()
    T1, T2 = ds.iloc[int(len(ds) * .8) - 1], ds.iloc[int(len(ds) * .9) - 1]
    tr = c[c.date <= T1]; va = c[(c.date > T1) & (c.date <= T2)]; te = c[c.date > T2]
    uv, pv, Htr = cohort(tr, va); ut, pt, _ = cohort(c[c.date <= T2], te)
    g = Htr.groupby('user_id').size()
    print(f'start={st}: review={len(c)} user={c.user_id.nunique()} T1={T1.date()} T2={T2.date()} train_user={Htr.user_id.nunique()} '
          f'RAM_userKNN={8*Htr.user_id.nunique()**2/1e9:.1f}GB val_users={uv} val_pos={pv} test_users={ut} test_pos={pt} '
          f'g1-4={int((g<=4).sum())} g5-19={int(((g>=5)&(g<=19)).sum())} g20+={int((g>=20).sum())}')
