import ezdxf

# ========================================================= #
# ===  make__sample_modelspace.py                       === #
# ========================================================= #

def make__sample_modelspace( outFile=None, radius=500.0, center=(250.,250.) ):

    # ------------------------------------------------- #
    # --- [1] parameters                            --- #
    # ------------------------------------------------- #

    # ------------------------------------------------- #
    # --- [2] prepare document &  modelspace        --- #
    # ------------------------------------------------- #
    doc       = ezdxf.new( dxfversion="R2018" )
    doc.units = ezdxf.units.MM
    msp       = doc.modelspace()
    

    msp.add_circle( center=center, radius=radius )
    if ( outFile ):
        doc.saveas( outFile )
        print( "[make__sample_modelspace.py] save in a file :: {} ".format( outFile ) )
    return( doc )

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    outFile = "dxf/original__modelspace.dxf"
    radius  = 500.0
    center  = ( 250., 250. )

    doc = make__sample_modelspace( outFile=outFile, radius=radius, center=center )
    print( doc )
    
    
