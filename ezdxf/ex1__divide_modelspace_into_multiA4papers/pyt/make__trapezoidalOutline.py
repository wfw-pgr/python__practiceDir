import os, sys
import ezdxf
import numpy as np

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
    doc.units     = ezdxf.units.MM
    msp           = doc.modelspace()

    # ------------------------------------------------- #
    # --- [3] trapezoidal vertex                    --- #
    # ------------------------------------------------- #
    vertex_lwrLef = [ 0.                      , 0.       ]
    vertex_lwrRgt = [ L_trapez_lwr            , 0.       ]
    vertex_uprRgt = [ L_trapez_lwr - halfDiff , H_trapez ]
    vertex_uprLef = [              + halfDiff , H_trapez ]
    vertices      = np.array( [ vertex_lwrLef, vertex_lwrRgt, \
                                vertex_uprRgt, vertex_uprLef, \
                                vertex_lwrLef ] )
    vertices      = vertices + 10.0

    # ------------------------------------------------- #
    # --- [4] randomly place circle                 --- #
    # ------------------------------------------------- #
    nCircle       = 50
    radius        = 10.0
    xrand         = np.random.uniform( size=(nCircle,1) )
    yrand         = np.random.uniform( size=(nCircle,1) )
    xLeng         = L_trapez_lwr * ( 1.0-yrand ) + L_trapez_upr * yrand
    xInit         = halfDiff * yrand
    xpos          = xInit + xLeng * xrand
    ypos          = H_trapez * yrand
    centers       = np.concatenate( [ xpos, ypos ], axis=1 )
    for ik, xy in enumerate( centers ):
        print( "(x,y) = ( {0[0]:10.3e}, {0[1]:10.3e} )".format( xy ) )
        circ = msp.add_circle( center=xy, radius=radius )
    
    
    # ------------------------------------------------- #
    # --- [5] draw trapezoidal model                --- #
    # ------------------------------------------------- #
    pl            = msp.add_lwpolyline( points=vertices, close=True )

    # ------------------------------------------------- #
    # --- [6] save in a file                        --- #
    # ------------------------------------------------- #
    outFile       = "dxf/trapezoid_example.dxf"
    doc.saveas( outFile )
    print( "[test05.py] output :: {} ".format( outFile ) )
    

    return()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    make__trapezoidalOutline()
