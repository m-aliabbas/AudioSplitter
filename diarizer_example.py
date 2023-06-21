from diarizer import Diarizer
import pandas as pd
input_filepath = "/home/idrak/Desktop/idrak_work/WhisperDiarizer/example_audios/20230320-095827_9095564802-all.wav"    
diarizer_obj = Diarizer()
tagged_segments = diarizer_obj.seprate_speakers(input_filepath)
df=pd.DataFrame(tagged_segments)
df.to_json('20230320-095827_9095564802-all.json')
print("*"*50)
print("Tagged Segments: \n", tagged_segments)
print("*"*50)