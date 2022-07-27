import speech_recognition as sr

# ========================================================= #
# ===  convert__soundData2text.py                       === #
# ========================================================= #

def convert__soundData2text( inpFile=None, outFile=None ):

    # ------------------------------------------------- #
    # --- [1] Argument check                        --- #
    # ------------------------------------------------- #
    if ( inpFile is None ): sys.exit( "[convert__soundData2text.py]  == ???" )
    if ( outFile is None ): outFile = "speech.txt"

    # ------------------------------------------------- #
    # --- [2] recognize speech in a sound file      --- #
    # ------------------------------------------------- #
    recog = sr.Recognizer()
    with sr.AudioFile( inpFile ) as source:
        audio = recog.record( source )
    text = recog.recognize_google( audio, language="ja" )
        
    print( "[convert__soundData2text.py] 音声データの文字起こし結果： \n\n", \
           recog.recognize_google( audio, language='ja' ) )
    print( "[convert__soundData2text.py] save in a file " )

    # ------------------------------------------------- #
    # --- [3] save in a file                        --- #
    # ------------------------------------------------- #
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            f.write( text )
        print( "[convert__soundData2text.py] save in a File :: {}".format( outFile ) )
    return( text )

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    inpFile = "wav/sound.wav"
    convert__soundData2text( inpFile=inpFile )
