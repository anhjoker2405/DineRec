# DineRec

Mã nguồn của khóa luận tốt nghiệp **"Xây dựng hệ thống khuyến nghị nhà hàng dựa trên lọc cộng tác"** và bài báo SSRC **"Xếp hạng lại theo khoảng cách cho khuyến nghị nhà hàng dựa trên lọc cộng tác: đánh giá theo thời gian trên dữ liệu Yelp với đối chứng độ phổ biến và vị trí"**.

- Tác giả: Lê Tuấn Anh, Nguyễn Minh Tiến, Lê Thị Vĩnh Thanh (giảng viên hướng dẫn)
- Khoa Công nghệ Thông tin, Trường Đại học Công nghiệp Thành phố Hồ Chí Minh, 2026

*English summary.* Code for a controlled offline comparison of collaborative filtering (UserKNN, ItemKNN, biased MF) against non-personalised baselines on Yelp restaurants in Indianapolis, with a post-hoc distance-aware re-ranking step, a popularity-plus-distance baseline (PopNear), a block-wise sparse Pearson implementation and a Flask + Leaflet prototype. The Yelp data are **not** included; see "Dữ liệu" below.

## Cấu trúc

| Thư mục / tệp | Nội dung |
|---|---|
| `code/prepare_data.py` | Đọc hai tệp JSON của Yelp, lọc nhà hàng Indianapolis, ghi `indy_reviews.csv` và `indy_business.csv` |
| `code/dinerec_core.py` | Pearson thưa theo khối, Haversine, vị trí đại diện, xếp hạng lại theo khoảng cách, PopNear |
| `code/run_experiments.py` | Toàn bộ thí nghiệm TN1–TN7: chia theo thời gian, tinh chỉnh trên Val, chọn mô hình nền và λ, đo trên Test, kiểm định |
| `code/ch3_analysis.py`, `code/ch3_extra.py` | Thống kê dữ liệu, khoảng tin cậy, các hình và số bổ sung của Chương 3 (không chọn lại tham số nào) |
| `code/tn8_real.py`, `code/plot_tn8.py` | TN8: so khớp với Surprise và đo bộ nhớ, thời gian |
| `code/lam_ref.py`, `code/tn9.py` | Phân tích độ nhạy của λ theo khoảng cách tham chiếu (mục 3.7 của khóa luận) và TN9 với tập ứng viên hợp nhất (mục 3.12); chỉ đọc các quyết định đã khóa, không chọn lại tham số nào |
| `code/fig_diagrams.py`, `code/paper_fig3.py` | Sơ đồ quy trình, sơ đồ xếp hạng lại, Hình 3 của bài báo |
| `code/vn_format.py` | Dấu phẩy thập phân cho mọi nhãn số trên hình |
| `prototype/` | Prototype web (Flask, Leaflet 1.9.4 đóng sẵn) và các script kiểm thử TN7 |
| `results/vong3/` | Các tệp kết quả tổng hợp của vòng tinh chỉnh 3 (số liệu được báo cáo). Không gồm tệp theo từng người dùng |

## Dữ liệu

Kho này **không** chứa dữ liệu Yelp, vì điều khoản sử dụng của Yelp không cho phân phối lại dữ liệu. Để chạy lại:

1. Tải **Yelp Open Dataset** tại <https://www.yelp.com/dataset> (nhóm tải ngày 05/11/2026; bản này có đánh giá mới nhất vào ngày 19/01/2022).
2. Giải nén, lấy hai tệp `yelp_academic_dataset_business.json` và `yelp_academic_dataset_review.json`, đặt chung một thư mục.

## Cài đặt

Python 3.13 trở lên.

```
pip install -r requirements.txt
```

Các phiên bản trong `requirements.txt` là môi trường đã chạy lại và cho kết quả trùng khớp (máy Linux 2 vCPU, RAM 7 GB).

## Chạy lại thực nghiệm (Chương 3 của khóa luận, Mục III–IV của bài báo)

Chạy trong thư mục `code`:

```
# 1. Xây dựng dữ liệu (in thêm thống kê của Bảng 3.2)
python prepare_data.py <thư mục chứa hai tệp JSON>

# 2. Vòng tinh chỉnh 3 (kết quả được báo cáo); khoảng 3 phút trên máy Linux 2 vCPU
python run_experiments.py --reviews indy_reviews.csv --business indy_business.csv --out results3 \
  --ks 5,10,20,40,80 --taus 2,5,10,20 --factors 20,50,100,200 \
  --regs 0.001,0.005,0.02,0.10 --epochs 10,20,40,80

# 3. Số liệu và hình bổ sung
python ch3_analysis.py indy_reviews.csv indy_business.csv results3 ch3
python ch3_extra.py indy_reviews.csv indy_business.csv results3 ch3

# 4. TN8 (mỗi lệnh là một tiến trình riêng để đo bộ nhớ đỉnh; MODE = check | ours | surprise, N = số người dùng, 0 = toàn bộ)
python tn8_real.py indy_reviews.csv indy_business.csv check 2000 5 5 >> tn8.jsonl
python tn8_real.py indy_reviews.csv indy_business.csv ours 0 5 5 >> tn8.jsonl
python plot_tn8.py tn8.jsonl ch3

# 5. Phân tích độ nhạy của λ (mục 3.7) và TN9 (mục 3.12); mỗi lệnh khoảng 15 giây
python lam_ref.py indy_reviews.csv indy_business.csv results3 results3/lam_ref.json
python tn9.py indy_reviews.csv indy_business.csv results3 results3/tn9.json
```

Vòng 1 và vòng 2 dùng lưới nhỏ hơn: vòng 1 là lệnh ở bước 2 với các giá trị mặc định (`--out results`), vòng 2 thêm `--out results2 --taus 2,5,10,20 --factors 20,50,100 --regs 0.005,0.02,0.10 --epochs 10,20,40`.

**Kiểm tra đã chạy đúng.** Tệp `results3/decisions_from_val.json` phải trùng với `results/vong3/decisions_from_val.json`: mô hình nền ItemKNN với k = 10, τ = 10, λ* = 0,2, λ_P = 0,1. Các tệp CSV phải trùng với các tệp trong `results/vong3/`, trừ cột thời gian chạy `sec`. Khi nhóm chạy lại trên một máy Linux khác, sai khác lớn nhất là 8,9 × 10⁻¹⁶ do thứ tự cộng số thực.

Hai tệp `lam_ref.json` và `tn9.json` cũng phải trùng với bản trong `results/vong3/`: λ** = 1 trên cả lưới thô và lưới mịn; TN9 chọn α* = 0,25, λ = 0,3 và đạt NDCG@10 = 0,0372 trên Test, hiệu so với PopNear 0,0012 [−0,0027; 0,0052]. TN9 và phân tích độ nhạy được thiết kế sau khi đã có kết quả chính, nên chỉ có giá trị thăm dò.

Mọi quyết định (cấu hình, mô hình nền, λ) được chọn trên tập kiểm định (Val). Tập Test chỉ dùng để báo cáo, nhưng đã được tính ở cả ba vòng tinh chỉnh; xem phần hạn chế trong khóa luận và bài báo.

## Prototype (Chương 4 của khóa luận)

Chạy trong thư mục `prototype`:

```
# Đóng gói mô hình (huấn luyện lại ItemKNN trên H_test với cấu hình đã khóa)
python build_model.py ../code/indy_reviews.csv ../code/indy_business.csv ../results/vong3 model

# Chạy máy chủ, rồi mở http://127.0.0.1:5000 (nền bản đồ cần Internet)
python app.py model

# TN7: so khớp API với quy trình ngoại tuyến và các ca kiểm thử
python tn7_api_check.py ../code/indy_reviews.csv ../code/indy_business.csv ../results/vong3 model tn7_api.json

# Thời gian phản hồi qua HTTP (trong lúc app.py đang chạy)
python tn7_http_check.py http://127.0.0.1:5000 model tn7_http.json
python ch4_numbers.py model tn7_api.json tn7_http.json ch4.json
```

Kết quả mong đợi của TN7: năm cấu hình trùng 100% với quy trình ngoại tuyến trên 1.823 người dùng, 18/18 ca kiểm thử đạt. Thời gian phản hồi phụ thuộc máy.

## Trích dẫn

Nếu dùng mã nguồn này, vui lòng trích dẫn bài báo SSRC 2026 của nhóm tác giả.

Leaflet 1.9.4 (`prototype/static/leaflet.js`, `leaflet.css`, `images/`) thuộc bản quyền của các tác giả Leaflet, phân phối theo giấy phép BSD 2-Clause (xem `prototype/static/LEAFLET-LICENSE.txt`).
