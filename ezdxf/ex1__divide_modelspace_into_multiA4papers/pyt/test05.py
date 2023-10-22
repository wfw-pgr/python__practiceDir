import os, sys
import ezdxf

# ========================================================= #
# ===  make__trapezoidOutline.py                        === #
# ========================================================= #

def make__trapezoidalOutline():
    
    w_, h_        = 0, 1
    
    # ------------------------------------------------- #
    # --- [1] parameter                             --- #
    # ------------------------------------------------- #
    L_trapez_lwr  = 600.0
    L_trapez_upr  = 450.0
    H_trapez      = 350.0
    halfDiff      = 0.5*( L_trapez_lwr - L_trapez_upr )

    # ------------------------------------------------- #
    # --- [2] generate .dxf model                   --- #
    # ------------------------------------------------- #
    doc           = ezdxf.new()
    msp           = doc.modelspace()

    # ------------------------------------------------- #
    # --- [3] trapezoidal vertex                    --- #
    # ------------------------------------------------- #
    vertex_lwrLef = [ 0.                      , 0.       ]
    vertex_lwrRgt = [ L_trapez_lwr            , 0.       ]
    vertex_uprRgt = [ L_trapez_lwr - halfDiff , H_trapez ]
    vertex_uprRgt = [              + halfDiff , H_trapez ]
    vertices      = [ vertex_lwrLef, vertex_lwrRgt, \
                      vertex_uprRgt, vertex_uprLef, \
                      vertex_lwrLef ]
    
    # ------------------------------------------------- #
    # --- [4] draw trapezoidal model                --- #
    # ------------------------------------------------- #
    pl            = msp.add_lwpolyline( points=vertices, close=True )

    # ------------------------------------------------- #
    # --- [5] save in a file                        --- #
    # ------------------------------------------------- #
    outFile       = "dxf/trapezoid_example.dxf"
    doc.saveas( outFile )
    print( "[test05.py] output :: {} ".format( outFile ) )
    

    return()
