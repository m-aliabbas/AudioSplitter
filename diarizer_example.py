from diarizer import Diarizer

input_filepath = "/home/mmb/Desktop/AfterGrad/IdrakWork/WhisperSplitter/audiosRecs/20220328-102350_8123448031-all.wav"    
diarizer_obj = Diarizer()
tagged_segments = diarizer_obj.seprate_speakers(input_filepath)
print("*"*50)
print("Tagged Segments: \n", tagged_segments)
print("*"*50)