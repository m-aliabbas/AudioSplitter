from pyannote.audio import Pipeline
import pandas as pd
import torchaudio
import os
import soundfile as sf
import json
from pathlib import Path
import sys
import whisper
from config import config

class Py_Diarizer:
    def __init__(self, auth_token) -> None: 
        self.model = whisper.load_model(config["model_size"])
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=auth_token)

    def get_pyannote_df(self, file_path, num_speakers=2):
        try:
            # Create an empty DataFrame
            df = pd.DataFrame(columns=['speaker_id', 'start', 'end'])
            
            # 4. apply pretrained pipeline
            diarization = self.pipeline(file_path)

            # 5. print the result
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # print(f"start={turn.start}s end={turn.end}s speaker_{speaker}")
                # print(type(turn.start))
                # start=0.2s stop=1.5s speaker_0
                # start=1.8s stop=3.9s speaker_1
                # start=4.2s stop=5.7s speaker_0
                # ...
                
                # Append a row of values
                new_row = {'speaker_id': speaker, 'start': turn.start, 'end': turn.end}
                df = df.append(new_row, ignore_index=True)
                
            return df
        except Exception as e:
            print("Error in pynote df function: ", e)
            
            
    def process_raw_time_stamps(self, df):
        try:
            updated_records =[]
            for k, v in df.groupby((df['speaker_id'].shift() != df['speaker_id']).cumsum()):
                # print(f'[group {k}]')
                startTime = v.start.values[0]
                endTime = v.end.values[-1]
                speaker_id = v.speaker_id.values[0]
                updated_records.append([speaker_id, startTime, endTime])

            df2 = pd.DataFrame(updated_records, columns=['speaker_id','start', 'end'])
            df2['turn'] = df2.groupby('speaker_id').cumcount().add(1)
            df2['chunk_id'] =  df2['speaker_id'].str.cat(df2['turn'].astype(str), sep="_")
            
            return df2

        except Exception as e:
            print("Error 70000: Process Raw Timestamps Fun", e)

    def delete_file(self, filepath):
        try:
            if os.path.isfile(filepath) == False:
                for file in os.listdir(filepath):
                    os.remove(os.path.join(filepath,file))
                os.rmdir(filepath)
                return None

            os.remove(filepath)
        except FileNotFoundError:
            print("Error, File not found")    
        except Exception as e:
            print("Error!", e)

    def createOutFolder(self, folderPath):
        try:
            Path(folderPath).mkdir(parents=True, exist_ok=True)
            return folderPath
        except Exception as e:
            print("Error", e)
            
            
    def audio_resample(self, input_filepath, required_samplerate=16000, output_filepath=None):
        """
        Changes the samplerate of an input audio file to the required samplerate.
        Stores the outputfile using the optional output_filepath, if None is give, overwrites the input file
        """
        waveform, samplerate = torchaudio.load(input_filepath)
        if samplerate != required_samplerate:
            waveform = torchaudio.functional.resample(waveform, samplerate, required_samplerate)
        if output_filepath is None:
            output_filepath = input_filepath
        torchaudio.save(output_filepath, waveform, required_samplerate, encoding="PCM_S", bits_per_sample=16)    


    def generate_splits_and_json(self, audio_filepath, df, fileId, outfolder_path, saveSplits = True):
        try:
            outfolder_path = self.createOutFolder(os.path.join(outfolder_path, fileId))
            splits_folder_path = self.createOutFolder(os.path.join(outfolder_path, "splits"))
            waveform, samplerate = torchaudio.load(audio_filepath)
            results_dic = dict()
            results_dic["file_id"] = fileId
            
            for i in range(len(df)):

                #Slicing chunk using time stamps
                start = df.loc[i,"start"]
                end = df.loc[i,"end"] 
                
                #Example
                #frame_offset, num_frames = 16000, 16000  # Fetch and decode the 1 - 2 seconds
                #waveform1 = waveform1[:, frame_offset : frame_offset + num_frames]               

                currentWaveform = waveform[:, int(start*samplerate):int(start*samplerate)+int((end-start)*samplerate)]
                currentFileName = df.loc[i,"chunk_id"]
                
                output_filepath = os.path.join(splits_folder_path, currentFileName+".wav")
                    
                # #Adding a pause at the end before saving the split
                # pause_length = 2 #seconds
                # pause_samples = torch.zeros((1,int(samplerate * pause_length)))
                # currentWaveform = torch.cat((currentWaveform, pause_samples), 1)
                
                #Saving chunk
                torchaudio.save(output_filepath, currentWaveform, samplerate, encoding="PCM_S", bits_per_sample=16)

                transcript = self.model.transcribe(output_filepath).get("text")
                
                results_dic[currentFileName] = {"trascript": transcript}
                
                
                #Deleting Splits folder if not needed
                if saveSplits == False:
                    self.delete_file(output_filepath)

            
            #saving results dictionary
            with open(os.path.join(outfolder_path,fileId+"_results.json"), "w") as outfile:
                json.dump(results_dic, outfile)

            #saving timestamps csv
            df.to_csv(os.path.join(outfolder_path,fileId+"_timestamps.csv"), index=False)

            return results_dic

        except Exception as e:
            print("Error 90000: ", e)

    def seprate_speakers(self, file_path, outfolder_path):
        
        if not os.path.exists(file_path):
            raise ValueError(f"{file_path} doesn't exist.")
        
        if not os.path.exists(outfolder_path):
            raise ValueError(f"Output folder path doesn't exist.")
        
        fileId, ext = os.path.splitext(os.path.basename(file_path))
        if ext != ".wav":
            raise ValueError(f"{ext} provided. Extension must be .wav")
        
        file_object = sf.SoundFile(file_path)
        if file_object.samplerate != 16000:
            self.audio_resample(file_path)
        
        df = self.get_pyannote_df(file_path)
        # print("Raw Timestamps: ", df) 

        df2 = self.process_raw_time_stamps(df)
        # print("Processed Timestamps: ", df2)

        results_dic = self.generate_splits_and_json(file_path, df2, fileId, outfolder_path)
        # print("Results Dictionary: ", results_dic)
        
        #saving raw timestamps
        df.to_csv(os.path.join(outfolder_path,fileId, fileId+"_raw_timestamps_.csv")) 
        
        return results_dic
    
def main():
    
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
    
if __name__ == "__main__":
    main()