from pyannote_diarizer import Py_Diarizer

file_path = "./junk/recent_testing/test_set/20220830-141812_27806095-all.wav"
outfolder_path = "OUTPUT"

whisper_model, auth_token = "small.en", "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"
diarizer = Py_Diarizer(auth_token)
print("\nProcessing...\n\n")
tagged_segments = diarizer.seprate_speakers(file_path, outfolder_path)
print("\nDone!")

print("*"*50)
print("Tagged Segments: \n", tagged_segments)
print("*"*50)