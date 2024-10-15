import os
from pydub import AudioSegment

# Directory containing MP3 files
input_dir = './'
# Directory to save WAV files
output_dir = 'wavs'

def mp32wav(input_dir:str, output_dir:str):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print(f"Files in {input_dir}:")
	print(os.listdir(input_dir))

	for file in os.listdir(input_dir):
		if file.endswith('.mp3'):
			filename = os.path.splitext(file)[0]
			mp3_path = os.path.join(input_dir, file)
			wav_path = os.path.join(output_dir, f"{filename}.wav")
			sound = AudioSegment.from_mp3(mp3_path)
			sound.export(wav_path, format='wav')
			print(f"Converted {file} to WAV")
	print("Conversion complete.")

def m4a2wav(input_dir:str, output_dir:str):
    for file in os.listdir(input_dir):
        if file.endswith('.m4a'):
            filename = os.path.splitext(file)[0]
            m4a_path = os.path.join(input_dir, file)
            wav_path = os.path.join(output_dir, f"{filename}.wav")
            
            # Load the M4A file
            sound = AudioSegment.from_file(m4a_path, format="m4a")
            
            # Export as WAV
            sound.export(wav_path, format='wav')
            print(f"Converted {file} to WAV")
    
    print("Conversion complete.")
    
m4a2wav('./', './')