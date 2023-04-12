
from whisper_splitter import Whisper_Splitter as diarizer

class Diarizer:
    def __init__(self) -> None: 
        self.diarizer = diarizer()

    def seprate_speakers(self, input_filepath):
        try:
            
            ext, filename, input_filepath, duration = self.diarizer.initial_processing(input_filepath)
            #Getting whisper results
            result = self.diarizer.get_whisper_result(input_filepath)
            segments = result["segments"]
            
            #Getting speaker tags
            tagged_segments = self.diarizer.get_speaker_tagged_segments(input_filepath, duration, segments)
            
            return tagged_segments
            
        except Exception as e:
            print("Error 10000!", e) 
               
def main():
    input_filepath = "/home/mmb/Desktop/AfterGrad/IdrakWork/WhisperSplitter/audiosRecs/20220328-102350_8123448031-all.wav"    
    diarizer_obj = Diarizer()
    tagged_segments = diarizer_obj.seprate_speakers(input_filepath)
    print("*"*50)
    print("Tagged Segments", tagged_segments)
    print("*"*50)
    
if __name__ == "__main__":
    main()

