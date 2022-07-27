import speech_recognition as sr

# ========================================================= #
# ===  recognize__sound2text.py                         === #
# ========================================================= #

def recognize__sound2text( inpFile=None, outFile=None ):

    # ------------------------------------------------- #
    # --- [1] Argument check                        --- #
    # ------------------------------------------------- #
    if ( inpFile is None ):
        print( "[recognize__sound2text.py] <  input file > " )
        print( "[recognize__sound2text.py]   .wav file convert :: $ ffmpeg -i ***.m4a -ab 256k output.wav etc.")
        print( "[recognize__sound2text.py] input  file name >>  ", end="" )
        inpFile = input()
    if ( outFile is None ):
        print( "[recognize__sound2text.py] < output file > " )
        print( "[recognize__sound2text.py] output file name >>  ", end="" )
        outFile = input()
    print( "[recognize__sound2text.py]  input File == {} ".format( inpFile ) )
    print( "[recognize__sound2text.py] output File == {} ".format( outFile ) )
        
    # ------------------------------------------------- #
    # --- [2] recognize speech in a sound file      --- #
    # ------------------------------------------------- #
    recog = sr.Recognizer()
    with sr.AudioFile( inpFile ) as source:
        audio = recog.record( source )
    text = recog.recognize_google( audio, language="ja" )
        
    print( "[recognize__sound2text.py] 音声データの文字起こし結果： \n\n", \
           recog.recognize_google( audio, language='ja' ) )
    print( "[recognize__sound2text.py] save in a file " )

    # ------------------------------------------------- #
    # --- [3] save in a file                        --- #
    # ------------------------------------------------- #
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            f.write( text )
        print( "[recognize__sound2text.py] save in a File :: {}".format( outFile ) )
    return( text )

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    recognize__sound2text()
