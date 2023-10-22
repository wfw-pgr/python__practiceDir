import os, sys
import ezdxf

# 新しいDXFファイルを作成
doc = ezdxf.new()

# モデルスペースを取得
msp = doc.modelspace()

# 台形の底辺の長さ
bottom_width = 450
top_width = 250

# 台形の高さ
height = 300

# 台形の頂点座標を計算
# left_top = (0, 0)
# right_top = (bottom_width, 0)
# right_bottom = (bottom_width - top_width, height)
# left_bottom = (top_width, height)

left_bottom = (0, 0)
right_bottom = (bottom_width, 0)
right_top = ( bottom_width - 0.5*(bottom_width-top_width), height)
left_top = ( + 0.5*(bottom_width-top_width), height)

# 台形を描画
msp.add_lwpolyline(points=[ left_top, right_top, right_bottom, left_bottom, left_top], close=True)

# DXFファイルに保存
doc.saveas('trapezoid_example.dxf')
