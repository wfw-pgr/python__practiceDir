import ezdxf

outFile = "dxf/test_out.dxf"
radius  = 500.0
center  = ( 250., 250. )

center1 = ( 100., 100. )
center2 = ( 400., 400. )
size1   = ( 500., 500. )

# 新しいDXFファイルを作成
doc = ezdxf.new()
msp = doc.modelspace()

msp.add_circle( center=center, radius=radius )

# ペーパースペース内にビューポートを作成
# vp = msp.add_viewport( center=center1, size=size1 )
insert = msp.add_blockref(name='*Paper_Space', insert=center1 )  # ペーパースペースの名前を指定
insert.dxf.insert = center2  # 新しい中心位置の座標

# # ペーパースペース内でのビューポートの中心位置を変更
# vp.dxf.center = center2  # 新しい中心位置の座標

# DXFファイルに保存
doc.saveas( outFile )
print( "[test.py] outFile :: {} ".format( outFile ) )
