import os, sys, math
import ezdxf
from   ezdxf import bbox
import numpy as np

# ========================================================= #
# ===  divide__modelspace_into_multiA4papers.py         === #
# ========================================================= #

def divide__modelspace_into_multiA4papers( inpFile=None, outFile=None, outBase=None, \
                                           dxfversion="R2018", page_width=None, page_height=None ):

    # -- coding index -- #
    w_, h_     = 0, 1

    # -- parameters -- #
    a4_width   = 210.0
    a4_height  = 297.0
    a3_width   = 297.0
    a3_height  = 420.0
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( inpFile     is None ): sys.exit("[divide__modelspace_into_multiA4papers.py] inpFile == ???")
    if ( outFile     is None ): outFile     = inpFile.replace( ".dxf", "_divided.dxf"       )
    if ( outBase     is None ): outBase     = outFile.replace( ".dxf", "_{0:02}_{1:02}.dxf" )
    if ( page_width  is None ): page_width  = a4_width
    if ( page_height is None ): page_height = a4_height
    
    # ------------------------------------------------- #
    # --- [2] import dxf file                       --- #
    # ------------------------------------------------- #
    doc        = ezdxf.readfile( inpFile )
    msp        = doc.modelspace()

    # ------------------------------------------------- #
    # --- [3] inquire boundigbox of the modelspace  --- #
    # ------------------------------------------------- #
    bb         = ezdxf.bbox.extents( msp, cache=ezdxf.bbox.Cache() )
    msp_width  = bb.extmax[w_] - bb.extmin[w_]
    msp_height = bb.extmax[h_] - bb.extmin[h_]
    msp_center = np.array( [ float( val ) for val in bb.center ] )
    print()
    print( "  modelspace_width  :: {}"                  .format( msp_width  ) )
    print( "  modelspace_height :: {}"                  .format( msp_height ) )
    print( "  modelspace_center :: ( {0[0]}, {0[1]} ) " .format( msp_center ) )
    print()

    # ------------------------------------------------- #
    # --- [4] reassemble bb by specified page size  --- #
    # ------------------------------------------------- #
    nPages_w     = math.ceil( msp_width  / page_width  )
    nPages_h     = math.ceil( msp_height / page_height )
    pageHalves   = np.array( [ 0.5*nPages_w*page_width, 0.5*nPages_h*page_height ] )
    bb_pages     = np.array( [ [ msp_center[w_]-pageHalves[w_], msp_center[h_]-pageHalves[h_] ],\
                               [ msp_center[w_]+pageHalves[w_], msp_center[h_]+pageHalves[h_] ] ] )
    page_centers = np.zeros( (nPages_w,nPages_h,2) )
    for ik in range( nPages_w ):
        for jk in range( nPages_h ):
            center_t = np.array( [ bb_pages[0][w_] + page_width *0.5 + float(ik)*page_width ,\
                                   bb_pages[0][h_] + page_height*0.5 + float(jk)*page_height ] )
            page_centers[ik,jk,:] = np.copy( center_t )
    print()
    print( "-"*60 )
    print( "   + nPages_w             :: {}".format( nPages_w ) )
    print( "   + nPages_h             :: {}".format( nPages_h ) )
    print( "-"*60 )
    print( "   + page_centers's shape :: {}".format( page_centers.shape ) )
    print( "-"*60 )
    print( "   + page_centers         :: [ " )
    for cnt in np.reshape( page_centers, (-1,2) ):
        print( "       ( {0[0]:10.3f}, {0[1]:10.3f} )"\
               .format( cnt ) )
    print( "      ] \n" )
    
    # ------------------------------------------------- #
    # --- [5] add viewport & output as a new file   --- #
    # ------------------------------------------------- #
    size   = ( page_width, page_height )
    center = ( 0.5*page_width, 0.5*page_height )
    for ik in range( nPages_w ):
        for jk in range( nPages_h ):
            center = page_centers[ik,jk,:]
            lo     = doc.layout()
            sp     = lo.page_setup( size=size, margins=(0,0,0,0) )
            vp     = lo.add_viewport( center=center, view_center_point=center, \
                                      size=size, view_height=size[1] )
            outFile_ = outBase.format( ik+1, jk+1 )
            doc.saveas( outFile_ )
            print( "[test06.py] save in a file :: {} ".format( outFile_ ) )

    # ------------------------------------------------- #
    # --- [6] return and end                        --- #
    # ------------------------------------------------- #
    return()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    inpFile = "dxf/original__modelspace.dxf"
    outFile = None
    divide__modelspace_into_multiA4papers( inpFile=inpFile, outFile=outFile )
