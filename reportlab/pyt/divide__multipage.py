import os, sys, math, json
import reportlab
import numpy                 as np
import draw__shapes          as dsh
from reportlab.lib           import pagesizes
from reportlab.lib.units     import mm
from reportlab.pdfgen        import canvas


# ========================================================= #
# ===  draw trayFrame                                   === #
# ========================================================= #

def draw__trayFrame( pdfcanvas=None, page_bbox=None, cards=None ):

    x_, y_ = 0, 1
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( pdfcanvas is None ): sys.exit( "[divide__multipage.py] pdfcanvas == ???" )
    if ( page_bbox is None ): sys.exit( "[divide__multipage.py] page_bbox == ???" )
    if ( cards     is None ): sys.exit( "[divide__multipage.py] cards     == ???" )

    # ------------------------------------------------- #
    # --- [2] draw shapes according to cards        --- #
    # ------------------------------------------------- #
    page_bbox_lb = np.copy( page_bbox[0:2] )
    for key, val in cards.items():
        # -- [2-1] line case                         -- #
        if ( val["shapeType"].lower() == "line"   ):
            startpt = ( val["start"] - page_bbox_lb )*mm
            endpt   = ( val["end"]   - page_bbox_lb )*mm
            pdfcanvas.line  ( startpt[x_], startpt[y_], endpt[x_], endpt[y_] )
        # -- [2-2] circle case                       -- #
        if ( val["shapeType"].lower() == "circle" ):
            centerpt = ( val["center"] - page_bbox_lb )*mm
            radius   = ( val["radius"] )*mm
            pdfcanvas.circle( centerpt[x_], centerpt[y_], radius )
            
        # -- [2-3] arc case                          -- #
        if ( val["shapeType"].lower() == "arc"    ):
            centerpt = val["center"] - page_bbox_lb
            x1, y1   = ( centerpt[x_]-val["radius"] )*mm, ( centerpt[y_]-val["radius"] )*mm
            x2, y2   = ( centerpt[x_]+val["radius"] )*mm, ( centerpt[y_]+val["radius"] )*mm
            a1, a2   = val["angle1"], val["angle2"] - val["angle1"]
            pdfcanvas.arc( x1, y1, x2, y2, a1, a2 )
            
    # ------------------------------------------------- #
    # --- [3] return pdf's canvas                   --- #
    # ------------------------------------------------- #
    return( pdfcanvas )


# ========================================================= #
# ===  draw shim hole                                   === #
# ========================================================= #

def draw__shimHole( pdfcanvas=None, page_bbox=None, holes=None ):

    x_, y_, r_ = 0, 1, 2
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( pdfcanvas is None ): sys.exit( "[divide__multipage.py] pdfcanvas == ???" )
    if ( page_bbox is None ): sys.exit( "[divide__multipage.py] page_bbox == ???" )
    if ( holes     is None ): sys.exit( "[divide__multipage.py] holes     == ???" )

    # ------------------------------------------------- #
    # --- [2] draw shapes according to cards        --- #
    # ------------------------------------------------- #
    page_bbox_lb = np.insert( np.array( page_bbox[0:2] ), 2, 0.0 )
    page_ref     = np.repeat( page_bbox_lb[np.newaxis,:], holes.shape[0], axis=0 )
    holes_page   = ( holes - page_ref ) * mm
    for xyr in holes_page:
        pdfcanvas.circle( xyr[x_], xyr[y_], xyr[r_] )
            
    # ------------------------------------------------- #
    # --- [3] return pdf's canvas                   --- #
    # ------------------------------------------------- #
    return( pdfcanvas )

    
    
    
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    x_ , y_ , r_ , f_  = 0, 1, 2, 3
    x1_, y1_, x2_, y2_ = 0, 1, 2, 3
    inpFile            = "dat/circularTray001.json"
    
    # ------------------------------------------------- #
    # --- [1] load settings                         --- #
    # ------------------------------------------------- #
    with open( inpFile, "r" ) as f:
        info = json.load( f  )
    cards    = info["shape"]
    settings = info["settings"]
    pagesize = np.array( settings["pagesize"] )

    # ------------------------------------------------- #
    # --- [2] load hole file                        --- #
    # ------------------------------------------------- #
    with open( settings["holeFile"], "r" ) as f:
        holes_ = np.loadtxt( f )
    flags = np.array( holes_[:,f_], dtype=bool )
    holes = ( holes_[ flags ] )[:,0:3]
        
    # ------------------------------------------------- #
    # --- [3] divide bbox into several pages        --- #
    # ------------------------------------------------- #
    overall_width    = ( settings["bbox"][x2_] - settings["bbox"][x1_] )
    overall_height   = ( settings["bbox"][y2_] - settings["bbox"][y1_] )
    nPage_w          = overall_width  / pagesize[x_]
    nPage_h          = overall_height / pagesize[y_]
    if ( not( nPage_w.is_integer() ) ): nPage_w = math.ceil( nPage_w )
    if ( not( nPage_h.is_integer() ) ): nPage_h = math.ceil( nPage_h )

    # ------------------------------------------------- #
    # --- [4] centering                             --- #
    # ------------------------------------------------- #
    if ( settings["centering"] ):
        overall_width_   = nPage_w * pagesize[x_]
        overall_height_  = nPage_h * pagesize[y_]
        margin_w         = 0.5 * ( overall_width_  - overall_width  )
        margin_h         = 0.5 * ( overall_height_ - overall_height )
        margins          = np.array( [ (-1.0)*margin_w, (-1.0)*margin_h, \
                                       (+1.0)*margin_w, (+1.0)*margin_h ] )
        tray_bbox        = np.array( settings["bbox"] ) + margins
    else:
        tray_bbox        = np.copy( settings["bbox"] )

    # ------------------------------------------------- #
    # --- [5] prepare page's origin points          --- #
    # ------------------------------------------------- #
    page_bbox = np.zeros( (nPage_w,nPage_h,4) )
    pagesize_ = np.concatenate( [pagesize,pagesize] )
    tray_base = np.concatenate( [tray_bbox[:2],tray_bbox[:2]] )
    for ik in range( nPage_w ):
        for jk in range( nPage_h ):
            page_bbox[ik,jk,:] = tray_base[:] + np.array( [ik,jk,ik+1,jk+1] ) * pagesize_[:]
            
    # ------------------------------------------------- #
    # --- [6] page settings for pdf canvas          --- #
    # ------------------------------------------------- #
    linewidth    = 1.0e-3
    outFileBase  = (settings["outFile"]).replace( ".pdf", "_{0:02}_{1:02}.pdf" )
    canvas_list  = []
    for ik in range( nPage_w ):
        for jk in range( nPage_h ):
            outFile            = outFileBase.format( ik+1,jk+1 )
            pdfcanvas = canvas.Canvas( outFile, pagesize=pagesizes.A3 )
            pdfcanvas.setLineWidth( linewidth )
            pdfcanvas = draw__trayFrame( pdfcanvas=pdfcanvas, \
                                         page_bbox=page_bbox[ik,jk,:], \
                                         cards=cards )
            pdfcanvas = draw__shimHole ( pdfcanvas=pdfcanvas, \
                                         page_bbox=page_bbox[ik,jk,:], \
                                         holes=holes )
            pdfcanvas.showPage()
            pdfcanvas.save()
            print( "[basic_sample.py] outFile :: {} ".format( outFile ) )
