from pyannote_diarizer import Py_Diarizer

file_path = "/home/idrak/Desktop/idrak_work/WhisperDiarizer/example_audios/20230320-095706_5635146427-all.wav"
outfolder_path = "OUTPUT"

whisper_model, auth_token = "base.en", "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"
diarizer = Py_Diarizer(auth_token)
print("\nProcessing...\n\n")
tagged_segments = diarizer.seprate_speakers(file_path, outfolder_path)
print("\nDone!")

print("*"*50)
print("Tagged Segments: \n", tagged_segments)
print("*"*50)