import os, sys
import numpy as np
import matplotlib.pyplot as plt


# ========================================================= #
# ===  matplotlib__inA4size.py                          === #
# ========================================================= #

def matplotlib__inA4size():

    x_, y_          = 0, 1
    xu_,yu_,xl_,yl_ = 0, 1, 2, 3
    
    # ------------------------------------------------- #
    # --- [1] figure settings                       --- #
    # ------------------------------------------------- #
    mm           = 1.0 / 25.4     #  1 inch = 25.4 mm
    figsize_mm   = np.array( [ 297, 210 ] )
    figsize_inch = figsize_mm * mm
    dpi          = 200
    radius       = 4
    pitch        = 11
    Nx, Ny       = 10, 10
    p0           = figsize_mm * 0.5
    nTheta       = 30
    bb           = [  0.0, 0.0, 1.0, 1.0 ]
    tMargin      = 16.0
    tFrom        =   0.0 + 0.5*tMargin
    tUntil       = 180.0 - 0.5*tMargin
    
    pngFile      = "png/out.png"
    epsFile      = "png/out.eps"
    pdfFile      = "png/out.pdf"

    # ------------------------------------------------- #
    # --- [2] generate data                         --- #
    # ------------------------------------------------- #
    theta_u     = np.linspace( tFrom, tUntil, nTheta ) * np.pi / 180.0 + 0.0
    theta_l     = np.linspace( tFrom, tUntil, nTheta ) * np.pi / 180.0 + np.pi
    xu0,yu0     = radius * np.cos( theta_u ), radius * np.sin( theta_u )
    xl0,yl0     = radius * np.cos( theta_l ), radius * np.sin( theta_l )
    xylu0       = np.concatenate( [xu0[:,None],yu0[:,None],xl0[:,None],yl0[:,None],], axis=1 )

    # ------------------------------------------------- #
    # --- [2] center position                       --- #
    # ------------------------------------------------- #
    xLen, yLen  = Nx*pitch, Ny*pitch
    xMin, xMax  = -0.5*xLen+p0[x_], +0.5*xLen+p0[x_]
    yMin, yMax  = -0.5*yLen+p0[y_], +0.5*yLen+p0[y_]
    import nkUtilities.equiSpaceGrid as esg
    x1MinMaxNum = [ xMin, xMax, Nx ]
    x2MinMaxNum = [ yMin, yMax, Ny ]
    x3MinMaxNum = [  0.0,  0.0,  1 ]
    coord       = ( esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                       x3MinMaxNum=x3MinMaxNum, returnType = "point" ) )[:,0:2]
    coord       = np.repeat( coord[:,None,:], nTheta, axis=1 )
    coord       = np.concatenate( [coord,coord], axis=2 )
    xylu0       = np.repeat( xylu0[None,:,:],  Nx*Ny, axis=0 )        #  [Nx*Ny,nTheta,4]
    xylu        = xylu0 + coord
    
    # ------------------------------------------------- #
    # --- [3] plot in figure                        --- #
    # ------------------------------------------------- #
    fig         = plt.figure( figsize=figsize_inch, dpi=dpi )
    ax          = fig.add_axes( [ bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1] ] )
    ax.set_xlim( 0.0, figsize_mm[0] )
    ax.set_ylim( 0.0, figsize_mm[1] )
    for ik, xy in enumerate( xylu ):
        ax.plot    ( xy[:,xu_], xy[:,yu_], color="black" )
        ax.plot    ( xy[:,xl_], xy[:,yl_], color="black" )
    plt.savefig( pdfFile, dpi=dpi )
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    matplotlib__inA4size()
