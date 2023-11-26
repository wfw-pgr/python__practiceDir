import os, sys
import googletrans
import nkTextProcess.extract__textFromPDF as etp

# ========================================================= #
# ===  translate__usingGoogleTrans.py                   === #
# ========================================================= #

def translate__usingGoogleTrans( inpFile=None, outFile=None, engFile=None, silent=True,
                                 ja_pdfFile=None ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( inpFile is None ): sys.exit( "[translate__usingGoogleTrans.py] inpFile == ???" )

    # ------------------------------------------------- #
    # --- [2] extract__textFromPDF                  --- #
    # ------------------------------------------------- #
    text_en = etp.extract__textFromPDF( inpFile=inpFile, outFile=engFile, remove_return=True )
    
    # ------------------------------------------------- #
    # --- [3] translate into japanese               --- #
    # ------------------------------------------------- #
    tr         = googletrans.Translator()
    text_stack = []
    for ik,apage in enumerate( text_en ):
        text_piece = ( tr.translate( apage, dest="ja", src="en" ) ).text
        text_stack+= [ text_piece ]
    text_ja = "\n\n".join( text_stack )

    # ------------------------------------------------- #
    # --- [4] save in a file                        --- #
    # ------------------------------------------------- #
    if ( not( silent ) ):
        print( "\n" + "-"*70 +"\n"  )
        print( text_ja )
        print( "\n" + "-"*70 +"\n"  )
        
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            f.write( text_ja )

    # ------------------------------------------------- #
    # --- [5] return                                --- #
    # ------------------------------------------------- #
    return( text_ja )
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    inpFile = "pdf/sample.pdf"
    outFile = "dat/text_ja.txt"
    engFile = "dat/text_en.txt"
    silent  = False
    translate__usingGoogleTrans( inpFile=inpFile, outFile=outFile, \
                                 engFile=engFile, silent=silent )



# buffLen    = 5000
# maxLen     = len( text_en ) - 1
# nBlock     = ( len( text_en ) // buffLen ) + min( 1, (len( text_en ))%buffLen )
# i1, i2     = (ik)*buffLen, min( (ik+1)*buffLen-1, maxLen )
# text_piece = ( tr.translate( text_en[i1:i2], dest="ja", src="en" ) ).text
    
