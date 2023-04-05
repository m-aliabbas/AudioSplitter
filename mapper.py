#!/usr/bin/env python3
import pandas as pd
from config import config
import torchaudio
import os
import json
import difflib

class Mapper:
    
    def similarity(self, orgPattern, genPattern):
        try:
            return difflib.SequenceMatcher(a=orgPattern.lower(), b=genPattern.lower()).ratio()
        except Exception as e:
            print("Error", e)                

    def check_response(self, patternsList, currentText):
        try:    
            bestMatch = ""
            bestSimilarity = 0.0
            bestMatchTag = ""
            for pattern in patternsList:
                patternOrg = pattern[0]
                patternTag = pattern[1]
                sim = self.similarity(patternOrg, currentText)
                if sim > bestSimilarity:
                    bestSimilarity = sim
                    bestMatch = currentText
                    bestMatchTag = patternTag
            
            if bestSimilarity > 0.7: #pattern found
                return True, bestSimilarity, bestMatch, bestMatchTag
            else:
                return False, bestSimilarity, bestMatch, "Not Pattern"
        except Exception as e:
            print("Error", e)                
                    
    def get_lookup_list(self, filename):
        try:
            # [[Pattern, Tag],...]
            # testList = [("hello", "Hello"), ("hi this is amy from american senior citizen’s care how are you doing today", "Introduction"), ("this call is about a new state regulated final expense insurance plan which covers a hundred percent of your burial funeral or cremation expenses it is specifically designed on people of fixed income or social security would you like to learn more about it", "InsurancePlan"), ("can you hear me", "CanYouHearMe"), ("to qualify you for the plan. Are you between the age of forty and eighty", "qualification"), ("say that again please", "SayThatAgainPlease") , ("I can bring my product specialist on the line and he can give you more information about it okey", "Transfer")]
            # return testList
            if filename[-4:] != ".csv":
                print("Error!, patterns.csv is missing")
                exit()
            df= pd.read_csv(filename)
            columns_titles = ["Text","Tag"]
            df=df.reindex(columns=columns_titles)
            df["Text"] = df["Text"].str.lower()
            reqList = df.values.tolist() # [[Pattern, Tag],...]
            return reqList   
            
        except Exception as e:
            print("Error 5000!,",e)

    def find_amy_speaker(self, df, avatar):
        bot = None
        speaker = None
        for i, row in df.iterrows():
            if avatar.lower() in row['text'].lower():
                bot = row['speaker_id']
                if speaker is not None:
                    return bot, speaker 
            else:
                speaker = row['speaker_id']
                if bot is not None:
                    return bot, speaker
        return None, None

    def map_responses(self, df, patternsList):
        try:
            mappedResponses = {}    
            bot, speaker = self.find_amy_speaker(df, config["avatar"])
            # print("-"*50)
            # print("Speaker", speaker)
            # print("Bot", bot)
            # print("-"*50)
            len_df = len(df)
            for i in range(len_df):
                currentText = df.loc[i,"text"]
                current_speaker = df.loc[i,"speaker_id"]
                found, bestSimilarity, bestMatch, bestMatchTag= self.check_response(patternsList, currentText)
                if found == True and current_speaker==bot:
                    chunk_id = df.loc[i,"chunk_id"]
                    start = df.loc[i,"start"]
                    end = df.loc[i,"end"]
                    mappedResponses[bestMatchTag] = {"bot_text": bestMatch, "bot_split_name": chunk_id} #Bot Sentense
                    next = i+1
                    if next < len_df:
                        currentText = df.loc[next,"text"]
                        current_speaker = df.loc[next,"speaker_id"]
                        
                        found, bestSimilarity, bestMatch, bestMatchTag2= self.check_response(patternsList, currentText)
                        if bestMatchTag2 == "Not Pattern" and current_speaker==speaker:
                            chunk_id = df.loc[next,"chunk_id"]
                            start = df.loc[next,"start"]
                            end = df.loc[next,"end"]
                            mappedResponses[bestMatchTag]["response"] = currentText #Response of Customer
                            mappedResponses[bestMatchTag]["response_split_name"] = chunk_id
                            mappedResponses[bestMatchTag]["start"] = start
                            mappedResponses[bestMatchTag]["end"] = end    
            
            return mappedResponses
        
        except Exception as e:
            print("Error 90000: ", e)    


if __name__ == "__main__":
    
    mapper = Mapper()
    path = "/home/mmb/Desktop/AfterGrad/IdrakWork/WhisperSplitter/test_modified.csv"
    df = pd.read_csv(path)

    filename = "20220328-102350_8123448031-all"
    audio_filepath = "/home/mmb/Desktop/AfterGrad/IdrakWork/WhisperSplitter/20220328-102350_8123448031-all.wav"
    outfolder_path = "/home/mmb/Desktop/AfterGrad/IdrakWork/WhisperSplitter/testResults/20220328-102350_8123448031-all"
    splits_folder_path = "/home/mmb/Desktop/AfterGrad/IdrakWork/WhisperSplitter/testResults/20220328-102350_8123448031-all/splits"
    pattern_filepath = "Amy_Patterns_Dec3rd.csv"
    patternsList = mapper.get_lookup_list(pattern_filepath)
    tags = [pattern[1] for pattern in patternsList] #Getting all tags from patterns list
    waveform, samplerate = torchaudio.load(audio_filepath)

    #getting mapped respones 
    mappedResponses = mapper.map_responses(df, patternsList)        
    print("Mapped Response: ", mappedResponses)

    #saving results dictionary
    with open(os.path.join(outfolder_path,filename+"_results.json"), "w") as outfile:
        json.dump(mappedResponses, outfile)

    #slicing the audio and saving chunks
    mapper.slice_save_responses(waveform, samplerate, mappedResponses, splits_folder_path)