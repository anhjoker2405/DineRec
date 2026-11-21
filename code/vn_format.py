"""Quy ước số tiếng Việt cho mọi hình: dấu phẩy thập phân (0,026; λ* = 0,2).
Chỉ cần `import vn_format` trước khi vẽ: mỗi lần savefig, nhãn trục, chú giải, chú thích và tiêu đề đều đổi
dấu chấm thập phân (chữ số.chữ số) sang dấu phẩy;
nhãn vạch chia là số nguyên từ 4 chữ số có dấu chấm phân cách hàng nghìn (3.500). Không đổi dữ liệu hay vị trí điểm."""
import re
import matplotlib.ticker as mt
from matplotlib.figure import Figure
from matplotlib.text import Text

_DEC = re.compile(r'(?<=\d)\.(?=\d)')


_INT = re.compile(r'^([−-]?)(\d{4,})$')


_GROUPED = set()   # các chuỗi do ng() tạo: dấu chấm ở đây là phân cách hàng nghìn, không được đổi


def vn(s):
    keep = sorted((g for g in _GROUPED if g in s), key=len, reverse=True)
    for k, g in enumerate(keep):
        s = s.replace(g, f'\x00{k}\x00')
    s = _DEC.sub(',', s)
    for k, g in enumerate(keep):
        s = s.replace(f'\x00{k}\x00', g)
    return s


def ng(n):
    """Số nguyên có dấu chấm phân cách hàng nghìn: 2005 -> 2.005."""
    g = f'{int(n):,}'.replace(',', '.')
    if '.' in g:
        _GROUPED.add(g)
    return g


def _tick(s):
    m = _INT.match(s)
    return m.group(1) + ng(m.group(2)) if m else vn(s)


class _Comma(mt.Formatter):
    def __init__(self, inner):
        self.inner = inner

    def set_axis(self, axis):
        super().set_axis(axis); self.inner.set_axis(axis)

    def set_locs(self, locs):
        self.locs = locs; self.inner.set_locs(locs)

    def __call__(self, x, pos=None):
        return _tick(self.inner(x, pos))

    def format_ticks(self, values):
        return [_tick(t) for t in self.inner.format_ticks(values)]

    def get_offset(self):
        return vn(self.inner.get_offset())


def _apply(fig):
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for get, put in ((axis.get_major_formatter, axis.set_major_formatter), (axis.get_minor_formatter, axis.set_minor_formatter)):
                f = get()
                if not isinstance(f, _Comma):
                    put(_Comma(f))
    for t in fig.findobj(Text):
        s = t.get_text()
        if s and _DEC.search(s):
            t.set_text(vn(s))


_orig = Figure.savefig


def _savefig(self, *a, **k):
    _apply(self)
    return _orig(self, *a, **k)


Figure.savefig = _savefig
