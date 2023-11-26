import os, sys
import googletrans
import nkTextProcess.extract__textFromPDF as etp
import nkTextProcess.convert__text2pdf    as t2p

# ========================================================= #
# ===  translator__usingGoogleTrans.py                  === #
# ========================================================= #

def translator__usingGoogleTrans( input_pdfFile=None, output_pdfFile=None, \
                                  english_txtFile=None, japanese_txtFile=None, silent=True ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( input_pdfFile  is None ): sys.exit("[translator__usingGoogleTrans.py] input_pdfFile  == ???")
    if ( output_pdfFile is None ): output_pdfFile = input_pdfFile.replace( ".pdf", "_ja.pdf" )

    # ------------------------------------------------- #
    # --- [2] extract__textFromPDF                  --- #
    # ------------------------------------------------- #
    text_en = etp.extract__textFromPDF( inpFile=input_pdfFile, outFile=english_txtFile, \
                                        remove_return=True )
    
    # ------------------------------------------------- #
    # --- [3] translator into japanese              --- #
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
        
    if ( japanese_txtFile is not None ):
        with open( japanese_txtFile, "w" ) as f:
            f.write( text_ja )

    # ------------------------------------------------- #
    # --- [5] convert into japanese pdf             --- #
    # ------------------------------------------------- #
    t2p.convert__text2pdf( outFile=output_pdfFile, texts=text_stack )
    
    # ------------------------------------------------- #
    # --- [6] return                                --- #
    # ------------------------------------------------- #
    return( text_ja )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    
    input_pdfFile    = "pdf/sample.pdf"
    output_pdfFile   = None
    english_txtFile  = "dat/text_en.txt"
    japanese_txtFile = "dat/text_ja.txt"
    silent           = True
    translator__usingGoogleTrans( input_pdfFile=input_pdfFile, output_pdfFile=output_pdfFile, \
                                  english_txtFile=english_txtFile, japanese_txtFile=japanese_txtFile,\
                                  silent=silent )



