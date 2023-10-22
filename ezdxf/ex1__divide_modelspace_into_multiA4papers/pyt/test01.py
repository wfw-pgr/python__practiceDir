import sys
import ezdxf

# parameters 
w_, h_    = 0, 1
a4_width  = 210
a4_height = 297

outFile   = "dxf/test_out.dxf"
radius    = 60

size      = ( a4_width, a4_height )
center    = ( 0.5*a4_width, 0.5*a4_height )

# 新しいDXFファイルを作成
doc       = ezdxf.new()
msp       = doc.modelspace()

# make model
msp.add_circle( center=center, radius=radius )

# Layout 
lo      = doc.layout()
sp      = lo.page_setup( size=size, margins=(0,0,0,0) )
vp      = lo.add_viewport( center=center, view_center_point=center, \
                           size=size, view_height=size[1] )

# DXFファイルに保存
doc.saveas( outFile )
print( "[test.py] outFile :: {} ".format( outFile ) )
