from pyannote_diarizer import Py_Diarizer

file_path = "./Filtered_97_audios/20220830-183132_2082095929-all.wav"
outfolder_path = "OUTPUT_P"

whisper_model, auth_token = "small.en", "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"
diarizer = Py_Diarizer(whisper_model, auth_token)
print("\nProcessing...\n\n")
tagged_segments = diarizer.seprate_speakers(file_path, outfolder_path)
print("\nDone!")

print("*"*50)
print("Tagged Segments: \n", tagged_segments)
print("*"*50)