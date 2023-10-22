import ezdxf

outFile = "dxf/test02.dxf"

# 新しいDXFファイルを作成
doc = ezdxf.new()

# ペーパースペースを取得
layout = doc.modelspace()

# 半径500mmの円を描画
center = (0, 0)
radius = 500
layout.add_circle(center, radius)

# ビューポートを作成
viewport = layout.add_viewport(center=(0, 0), size=(250, 250))

# ビューポート内に表示するコンテンツを指定
viewport.dxf.view_center_point = (0, 0)  # ビューポート内の表示中心位置
viewport.dxf.view_height = 250  # 表示領域の高さ

# DXFファイルに保存
doc.saveas( outFile )
