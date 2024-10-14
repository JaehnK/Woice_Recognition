import speech_recognition as sr

def SpeechRecognition(AudioPath: str) -> str:
    r = sr.Recognizer()
    WavFile = sr.AudioFile(AudioPath)
    print("Comparing Words...")
    with WavFile as source:
        audio = r.record(source, duration=5)
    return r.recognize_google(audio_data=audio, language='ko-KR')


def main() -> int:
    print("Compare")
    audio = '/home/jaehun/redimnet/hb/hyebin.wav'
    print(SpeechRecognition(audio))

if __name__ == "__main__":
    main()
