    #!/usr/bin/env python3
import torchaudio
import os
import whisper
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
import numpy as np  
import pandas as pd
from config import config
import soundfile as sf
from pathlib import Path
import json
from datetime import datetime, timedelta
from mapper import Mapper


class Whisper_Splitter:
    
    def __init__(self) -> None: 
        self.embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
        self.model = whisper.load_model(config["model_size"])
        self.audio = Audio()
        self.mapper = Mapper()
    
    def createOutFolder(self, folderPath):
        try:
            Path(folderPath).mkdir(parents=True, exist_ok=True)
            return folderPath
        except Exception as e:
            print("Error", e)
    
    def mp3_2_wav(self, input_filepath, output_filepath=None):
        filename, ext = os.path.splitext(input_filepath)
        if output_filepath is None:
            output_filepath = filename + ".wav"
        
        # waveform, samplerate = torchaudio.load(input_filepath)
        # torchaudio.save(output_filepath, waveform, samplerate, encoding="PCM_S", bits_per_sample=16)
        subprocess.call(['ffmpeg', '-i', input_filepath, output_filepath, '-y'])
  
        
        return output_filepath
    
    def get_stats(self, file_path):
        waveform, samplerate = torchaudio.load(file_path)

        # Get audio duration
        duration = float(waveform.shape[1]) / samplerate

        # Get number of frames
        num_frames = waveform.shape[1]

        return duration, num_frames, samplerate
        
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

    def get_whisper_result(self, file_path):
        result = self.model.transcribe(file_path)
        return result
    
    def segment_embedding(self, file_path, duration, segment):
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = self.audio.crop(file_path, clip)
        return self.embedding_model(waveform[None])

    def get_speaker_tagged_segments(self, file_path, duration, segments):
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = self.segment_embedding(file_path, duration, segment)
        embeddings = np.nan_to_num(embeddings)
        
        # clustering = SpectralClustering(n_clusters=config['num_speakers'], affinity='nearest_neighbors').fit(embeddings)
        # clustering = KMeans(n_clusters=config['num_speakers']).fit(embeddings)
        
        clustering = AgglomerativeClustering(config['num_speakers']).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
        
        return segments

    def time(self, secs):
        return timedelta(seconds=round(secs, 3))

    def get_speaker_tagged_df(self, segments):
        # Create an empty DataFrame
        df = pd.DataFrame(columns=['speaker_id', 'start', 'end', 'text'])
        for segment in segments:
            speaker = segment["speaker"]
            start_time = str(self.time(segment["start"]))
            end_time = str(self.time(segment["end"]))
            text = segment["text"][1:] 
                    
            # Append a row to the DataFrame
            new_row = {'speaker_id': speaker, 'start': start_time , 'end': end_time, 'text': text}
            df = df.append(new_row, ignore_index=True)    
        
        return df
        
    def save_results(self, df, file_path="results.csv"):
        df.to_csv(file_path, index=False)
    
    
    def merge_text_by_speaker(self, df):
        # initialize variables for grouping
        speaker_id = None
        start = None
        end = None
        text = None
        grouped_rows = []

        # iterate over the rows in the original dataframe
        for index, row in df.iterrows():
            # if the current row has the same speaker_id as the previous row, concatenate the text
            if row['speaker_id'] == speaker_id:
                text += ' ' + row['text']
                end = row['end']
            # otherwise, add the previous row to the list and start a new group
            else:
                if speaker_id is not None:
                    grouped_rows.append({'speaker_id': speaker_id,
                                        'start': start,
                                        'end': end,
                                        'text': text})
                speaker_id = row['speaker_id']
                start = row['start']
                end = row['end']
                text = row['text']
                
        # add the last row to the list
        grouped_rows.append({'speaker_id': speaker_id,
                            'start': start,
                            'end': end,
                            'text': text})

        # create a new dataframe from the grouped rows
        df2 = pd.DataFrame(grouped_rows)

        return df2
        
    def processRawTimeStamps(self, df):
        try:
            #dropping nan columns
            df.dropna(axis=1, how='all', inplace=True)

            df2 = self.merge_text_by_speaker(df)
            
            # replace spaces with underscores in the speaker_id column
            df2['speaker_id'] = df2['speaker_id'].str.replace(' ', '_')
            
            df2['turn'] = df2.groupby('speaker_id').cumcount().add(1)

            df2['chunk_id'] =  df2['speaker_id'].str.cat(df2['turn'].astype(str), sep="_")
            # print(df2)

            return df2

        except Exception as e:
            print("Error 5000: Process Timestamps Fun", e)
    
    def time_to_seconds(self, time_str):
        try:
            if '.' in time_str:
                format_str = '%H:%M:%S.%f'
            else:
                format_str = '%H:%M:%S'
               
            time_obj = datetime.strptime(time_str, format_str)
            time_delta = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second, microseconds=time_obj.microsecond)
            return time_delta.total_seconds()
        
        except Exception as e:
            print("-"*50)
            print(time_str)
            print(type(time_str))
            print("-"*50)
            print(e)
            print("-"*50)
        
    def slice_save_responses(self, waveform, samplerate, mappedResponses, folder_path):
        for tag in mappedResponses.keys():
            if "response" in mappedResponses[tag].keys():
                start = self.time_to_seconds(mappedResponses[tag]["start"])
                end = self.time_to_seconds(mappedResponses[tag]["end"])
                # print(start, end)
                # print(type(start), type(end))
                
                currentWaveform = waveform[:, int(start*samplerate):int(start*samplerate)+int((end-start)*samplerate)]
                filename = tag+".wav"
                output_filepath = os.path.join (folder_path, filename)
                
                #Saving chunk
                torchaudio.save(output_filepath, currentWaveform, samplerate, encoding="PCM_S", bits_per_sample=16)
    
    
    def initial_processing(self, input_filepath):
        basename = os.path.basename(input_filepath)
        splitted_name = basename.split('.')
        filename, ext = splitted_name[0], splitted_name[1]
        # -- Converting mp3 to wav if required
        if ext != "wav":
            input_filepath = self.mp3_2_wav(input_filepath)

        # -- Resampling if reuired
        file_object = sf.SoundFile(input_filepath)
        if file_object.samplerate != 16000:
            self.audio_resample(input_filepath)
            file_object = sf.SoundFile(input_filepath)

    
        # Get audio duration
        duration = float(len(file_object)) / file_object.samplerate

        # Get number of frames
        num_frames = file_object.frames
        
        return ext, filename, input_filepath, duration
        
    
    def split_audio_file(self, input_filepath, output_dir_path, patternsList, verbose):
        try:
            #Dict for storing results
            results_dic = dict()
            
            ext, filename, input_filepath, duration = self.initial_processing(input_filepath)
            results_dic["id"] = filename
            
            #Getting whisper results
            result = self.get_whisper_result(input_filepath)
            segments = result["segments"]
            
            #Getting speaker tags
            tagged_segments = self.get_speaker_tagged_segments(input_filepath, duration, segments)
            
            df = self.get_speaker_tagged_df(tagged_segments)
            
            #Grouping and processing raw time stamps
            df_updated = self.processRawTimeStamps(df)
            
            #Saving transcript 
            results_dic["transcript"] = result["text"]

            #Creating output folder for audio file under processing
            outfolder_path = os.path.join(output_dir_path,filename)
            self.createOutFolder(outfolder_path)
            
            #Creating sub folder (splits) in output folder
            splits_folder_path = os.path.join(outfolder_path,"splits/")
            self.createOutFolder(splits_folder_path)
            
            #Loading audio 
            waveform, samplerate = torchaudio.load(input_filepath)

            #Getting mapped respones 
            mappedResponses = self.mapper.map_responses(df_updated, patternsList)        
            results_dic["responses"] = mappedResponses
            # print("Mapped Response: ", mappedResponses)

            #Saving intermetiate results for testing/debugging
            self.save_results(df, file_path=os.path.join(outfolder_path,filename+"_whisper.csv"))
            self.save_results(df_updated, file_path=os.path.join(outfolder_path,filename+"_whisper_updated.csv"))
            
            #Saving results dict
            with open(os.path.join(outfolder_path,filename+".json"), "w") as outfile:
                json.dump(results_dic, outfile)

            #slicing the audio and saving chunks
            self.slice_save_responses(waveform, samplerate, mappedResponses, splits_folder_path)
            
            #Deleting the newly generated wav file in case mp3 was given
            if ext != "wav":
                os.remove(input_filepath)  
        except Exception as e:
            print("Error 10000!", e) 
               
def main():    
    pass
    
if __name__ == "__main__":
    main()

