import speech_recognition as sr
import os
import json

def LoadEmbeddings(name: str) -> dict:
    embed_path = os.path.join('./voice_embeddings', f'{name}_metadata.json')
    # 파일이 존재하는지 확인
    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"No metadata found for {name} at {embed_path}")

    # 파일 읽기 및 JSON 파싱
    with open(embed_path, 'r') as f:
        embeddings = json.load(f)
    
    # print(f"Loaded embeddings for {name}: {type(embeddings)}")  # 딕셔너리 확인
    # print(embeddings.keys())
    return embeddings

def SpeechRecognition(AudioPath: str, name: str) -> str:
    embeddings = LoadEmbeddings(name)
    code = embeddings['secret_code']
    lang = embeddings['code_language']
    r = sr.Recognizer()
    WavFile = sr.AudioFile(AudioPath)
    # print("Comparing Words...")
    with WavFile as source:
        audio = r.record(source, duration=5)
    InputCode = r.recognize_google(audio_data=audio, language=lang)
    # print(InputCode, code)
    if (code == InputCode):
        print("Driver Authentification Passed Welcome")
        print(f"\033[1;34mEnjoy Your Journey {name}\033[0m")
    else:
        print("Driver Authentification Failed. ")
    
    return 


# def main() -> int:
#     print("Compare")
#     audio = '/home/jaehun/redimnet/hb/hyebin.wav'
#     print(SpeechRecognition(audio))

# if __name__ == "__main__":
#     LoadEmbeddings('hyebin')
#     SpeechRecognition('./sample_voices/1.wav', 'hyebin')
