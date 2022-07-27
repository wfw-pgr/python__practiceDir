import speech_recognition as sr

AUDIO_FILE = "wav/sound.wav" #ここを変更。アップロードした音声ファイル（.wav形式）名に変更してください。

# ========================================================= #
# ===  convert__soundData2text.py                       === #
# ========================================================= #



r = sr.Recognizer()
with sr.AudioFile( AUDIO_FILE ) as source:
    audio = r.record(source)

print('音声データの文字起こし結果：\n\n', r.recognize_google(audio, language='ja'))
