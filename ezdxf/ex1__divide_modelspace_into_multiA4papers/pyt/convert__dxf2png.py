import sys, os
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# -- [howto] -- #
#
# $ python pyt/convert__dxf2png.py  "inpFile"
#
# -- [howto] -- #

# ------------------------------------------------- #
# --- [1] input / output                        --- #
# ------------------------------------------------- #

if ( len( sys.argv ) >= 2 ):
    inpFile = str( sys.argv[1] )
else:
    # inpFile = "dxf/original__modelspace.dxf"
    inpFile = "dxf/divided__paperspace.dxf"

outFile = inpFile.replace( "dxf", "png" )

print()
print( "[convert__dxf2png.py] inpFile :: {} ".format(inpFile) )
print( "[convert__dxf2png.py] outFile :: {} ".format(outFile) )
print()

doc = ezdxf.readfile( inpFile )
msp = doc.modelspace()

fig, ax = plt.subplots()
ctx     = RenderContext(doc)
out     = MatplotlibBackend(ax)
front   = Frontend( ctx, out )
Frontend(ctx, out).draw_layout(msp, finalize=True)
# front.render( msp, layout=doc.modelspace() )

# 保存
fig.savefig( outFile, dpi=300 )  # ファイル名とdpiを設定
