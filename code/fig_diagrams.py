"""Vẽ lại sơ đồ Hình 2.1 (quy trình đánh giá) và Hình 2.3 (xếp hạng lại theo khoảng cách).
Dùng: python fig_diagrams.py OUT_DIR
(Sơ đồ không có số thập phân nên không cần vn_format; số hiệu bảng, công thức giữ dấu chấm.)"""
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
FILL = dict(grey='#eeeeee', white='#ffffff', yellow='#fff2cc', red='#f4cccc', green='#d9ead3', blue='#cfe2f3', purple='#e6dcef')


def box(ax, x, y, w, h, text, fill='white', fs=9, bold=False, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12', fc=FILL[fill], ec='#222222', lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color='#222222', ls='-'):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=12))


def stage(ax, x, y, w, h, title, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0', fc=col + '22', ec=col, lw=1.4, ls=(0, (4, 3))))
    ax.text(x + 0.15, y + h - 0.35, title, fontsize=FS + 1, fontweight='bold', color=col, va='center')


# ============================== Hình 2.1: quy trình đánh giá theo thời gian
FS = 8.5
fig, ax = plt.subplots(figsize=(9.2, 6.6)); ax.set_xlim(0, 16); ax.set_ylim(1.7, 12); ax.axis('off')
box(ax, 0.3, 10.6, 4.6, 1.2, 'Yelp Open Dataset\nbusiness, review', 'grey', FS)
box(ax, 5.6, 10.6, 4.8, 1.2, 'Lọc nhà hàng, tọa độ,\nđiểm 1–5, ngày hợp lệ', 'white', FS)
box(ax, 11.1, 10.6, 4.6, 1.2, 'Khử trùng, giữ đánh giá\nđầu tiên mỗi cặp u–i', 'white', FS)
arrow(ax, 4.9, 11.2, 5.6, 11.2); arrow(ax, 10.4, 11.2, 11.1, 11.2)
box(ax, 3.8, 8.85, 8.4, 0.95, 'Chia theo hai mốc T₁, T₂ → Train, Val, Test', 'yellow', FS)
ax.plot([13.4, 13.4], [10.6, 9.33], color='#222222', lw=1.3); arrow(ax, 13.4, 9.33, 12.2, 9.33)
stage(ax, 0.3, 3.9, 7.4, 4.55, 'Giai đoạn 1 ($H_{val}$ = Train)', '#3d6db5')
stage(ax, 8.3, 3.9, 7.4, 4.55, 'Giai đoạn 2 ($H_{test}$ = Train ∪ Val)', '#7d4f9a')
arrow(ax, 6.0, 8.85, 4.0, 8.45); arrow(ax, 10.0, 8.85, 12.0, 8.45)
box(ax, 0.6, 6.75, 6.8, 1.05, 'Lọc $n_{min}$ trên $H_{val}$; tính μ, tương đồng,\nphổ biến, vị trí đại diện từ $H_{val}$', 'white', FS)
box(ax, 0.6, 5.4, 6.8, 1.05, 'Huấn luyện bốn đối chứng (Bảng 2.1),\nUserKNN, ItemKNN và Biased MF', 'white', FS)
box(ax, 0.6, 4.1, 6.8, 1.05, 'Đo trên Val: NDCG@10 → chọn cấu hình,\nmô hình nền, λ* và $λ_P$ (PopNear)', 'yellow', FS)
arrow(ax, 4.0, 6.75, 4.0, 6.45); arrow(ax, 4.0, 5.4, 4.0, 5.15)
box(ax, 8.6, 6.75, 6.8, 1.05, 'Tính lại mọi thống kê, danh mục và\nvị trí đại diện từ $H_{test}$', 'white', FS)
box(ax, 8.6, 5.4, 6.8, 1.05, 'Huấn luyện lại với các lựa chọn đã khóa;\nxếp hạng lại với λ*, PopNear với $λ_P$', 'white', FS)
box(ax, 8.6, 4.1, 6.8, 1.05, 'Đo trên Test để báo cáo, không dùng để chọn:\nP, R, NDCG, RMSE, MAE, Coverage, km', 'red', FS)
arrow(ax, 12.0, 6.75, 12.0, 6.45); arrow(ax, 12.0, 5.4, 12.0, 5.15)
box(ax, 0.6, 1.9, 6.8, 1.15, 'Khóa: siêu tham số, mô hình nền, λ*, $λ_P$\n(không dùng kết quả Test để chọn)', 'green', FS)
arrow(ax, 4.0, 4.1, 4.0, 3.05)
ax.plot([7.4, 8.0, 8.0], [2.47, 2.47, 5.92], color='#2e6b1f', lw=1.3); arrow(ax, 8.0, 5.92, 8.6, 5.92, color='#2e6b1f')
box(ax, 8.6, 1.9, 6.8, 1.15, 'Lưu mô hình, ánh xạ ID, vị trí đại diện\nvà manifest cho prototype', 'grey', FS)
arrow(ax, 12.0, 4.1, 12.0, 3.05)
fig.savefig(f'{out}/fig21_pipeline.png'); plt.close(fig)

# ============================== Hình 2.3: các bước xếp hạng lại
FS = 8.5
fig, ax = plt.subplots(figsize=(9.2, 4.6)); ax.set_xlim(0, 16); ax.set_ylim(0, 8.4); ax.axis('off')
box(ax, 0.2, 6.2, 3.4, 1.6, 'Lịch sử của u\n→ vị trí đại diện\n(trung bình tọa độ)', 'grey', FS)
box(ax, 0.2, 2.9, 3.4, 1.6, 'Mô hình nền:\nđiểm dự đoán $\\hat{r}_{ui}$\ncho $i \\in C_u$', 'blue', FS)
box(ax, 4.1, 2.6, 4.9, 2.2, 'Lấy M ứng viên theo khóa (2.8):\n$\\hat{r}_{ui}$ chưa cắt, giảm dần;\nnhà hàng không có láng giềng\nxếp sau → $C_u^M$ (M > K)', 'white', FS)
box(ax, 4.1, 6.2, 4.9, 1.6, 'Haversine: khoảng cách\n$d_{ui}$ (km) tới mọi $i \\in C_u^M$', 'white', FS)
box(ax, 9.45, 4.0, 3.3, 2.6, 'Min–max trên $C_u^M$:\n$s_{ui}$ (sở thích; = 0 nếu\nkhông có láng giềng)\n$g_{ui}$ (gần)', 'yellow', FS)
box(ax, 13.1, 4.5, 2.8, 1.6, 'điểm =\n$λ s_{ui} + (1-λ) g_{ui}$', 'yellow', FS)
box(ax, 13.0, 1.2, 3.0, 2.0, 'Top-K theo điểm;\nhòa: có láng giềng\ntrước, $\\hat{r}_{ui}$ giảm,\nrồi business_id', 'green', FS)
arrow(ax, 3.6, 7.0, 4.1, 7.0); arrow(ax, 3.6, 3.7, 4.1, 3.7); arrow(ax, 6.55, 4.8, 6.55, 6.2)
arrow(ax, 9.0, 7.0, 9.45, 6.1); arrow(ax, 9.0, 3.7, 9.45, 4.5); arrow(ax, 12.75, 5.3, 13.1, 5.3); arrow(ax, 14.5, 4.5, 14.5, 3.2)
ax.text(0.2, 0.7, 'λ = 1: giữ nguyên thứ tự của mô hình nền; λ giảm: ưu tiên nhà hàng gần hơn. '
        'Điểm chưa cắt dùng để xếp hạng;\nphép cắt về [1, 5] chỉ dùng khi tính RMSE, MAE.', fontsize=FS - 0.5, va='center')
fig.savefig(f'{out}/fig23_rerank.png'); plt.close(fig)

# ============================== Hình 1 của bài báo: bản một cột của Hình 2.1
FS = 6.5
fig, ax = plt.subplots(figsize=(3.4, 4.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 11.6); ax.axis('off')
box(ax, 0.1, 10.3, 9.8, 1.1, 'Yelp Open Dataset: lọc, khử trùng\n(giữ đánh giá đầu tiên mỗi cặp u–i)', 'grey', FS)
box(ax, 0.1, 8.75, 9.8, 0.95, 'Chia theo hai mốc $T_1$, $T_2$ → Train, Val, Test', 'yellow', FS)
arrow(ax, 5.0, 10.3, 5.0, 9.7)
stage(ax, 0.1, 2.35, 4.75, 6.0, 'Giai đoạn 1: $H_{val}$', '#3d6db5')
stage(ax, 5.15, 2.35, 4.75, 6.0, 'Giai đoạn 2: $H_{test}$', '#7d4f9a')
arrow(ax, 4.3, 8.75, 4.3, 8.35); arrow(ax, 5.7, 8.75, 5.7, 8.35)
box(ax, 0.3, 5.9, 4.35, 1.5, 'Thống kê từ $H_{val}$;\nhuấn luyện mô hình', 'white', FS)
box(ax, 0.3, 3.95, 4.35, 1.5, 'Đo trên Val: chọn\ncấu hình, mô hình\nnền, λ*, $λ_P$', 'yellow', FS)
box(ax, 5.35, 5.9, 4.35, 1.5, 'Thống kê lại từ $H_{test}$;\nhuấn luyện lại', 'white', FS)
box(ax, 5.35, 3.95, 4.35, 1.5, 'Đo trên Test:\nchỉ báo cáo,\nkhông dùng để chọn', 'red', FS)
arrow(ax, 2.47, 5.9, 2.47, 5.45); arrow(ax, 7.52, 5.9, 7.52, 5.45)
box(ax, 0.3, 2.55, 4.35, 0.95, 'Khóa mọi lựa chọn', 'green', FS)
arrow(ax, 2.47, 3.95, 2.47, 3.5)
ax.plot([4.65, 5.0, 5.0], [3.02, 3.02, 6.65], color='#2e6b1f', lw=1.1); arrow(ax, 5.0, 6.65, 5.35, 6.65, color='#2e6b1f')
box(ax, 0.1, 0.2, 9.8, 1.3, 'Kết quả trên Test; đóng gói mô hình\nvà vị trí đại diện cho prototype', 'grey', FS)
arrow(ax, 7.52, 3.95, 7.52, 1.5)
fig.savefig(f'{out}/fig21_pipeline_motcot.png'); plt.close(fig)
print('ok', out)
