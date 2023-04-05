import os
from pathlib import Path
from tqdm import tqdm
from mapper import Mapper
import pandas as pd
from whisper_splitter import Whisper_Splitter
from config import config

def readInputFolder(inpDirName):
    try:
        fileList = os.listdir(inpDirName)
        leniF = len(fileList)
        if leniF == 0:
            print("Error! Input Folder is empty. Kindly add files in input folder.")
            exit()
        return fileList         
    except Exception as e:
        print(e)
        exit()

def checkFileFormat(inpFileName):
    """
    This function checks for the format (correct extension) of the input file. 
    input format could be mp3, wav etc 

    Args:
        inpFileName (str): Name of the file that needs to be processed.

    Returns:
        'Format' (str): e.g, in case file is mp3 then it will return mp3
    """
    try:
        root, extention = os.path.splitext(inpFileName)
        return extention
      
    except Exception as e:
        print("Error", e)
        exit()

def macFolderProblem(myList):
    try:
        if ".DS_Store" in myList:
            myList.remove(".DS_Store")
        return myList
    except Exception as e:
        print("Error", e)
        exit()

def createOutFolder(inFolder, outFolder):
    try:
        outPath = os.path.join(outFolder,inFolder)
        Path(outPath).mkdir(parents=True, exist_ok=True)
        return outPath
    except Exception as e:
        print("Error", e)
        exit()
                
def main():
    print(''.center(40,'*'))
    print('Whisper Splitter Tool version 1.0'.center(40))
    print('Copyright (C) IDRAK AI'.center(40))
    print('Created by M Musawar'.center(40))
    print('Date: 5th April 2023'.center(40))
    print(''.center(40,'*'))
    
    try:
        verbose = True
        inp_dir = "audiosRecs"
        output_dir = "OUTPUT"
        
        pattern_filepath = config["patterns_path"]
        cwd = os.getcwd()
        inp_dir_path = os.path.join(cwd,inp_dir)
        output_dir_path = os.path.join(cwd,output_dir)
        
        #Getting Audios List
        inFileList = readInputFolder(inp_dir)
        inFileList = macFolderProblem(inFileList)
        
        #Creating Output Folder 
        output_dir_path = createOutFolder(inp_dir, output_dir_path)    
        
        if verbose == True:
            print("Loading Models...")
        
        splitter = Whisper_Splitter()
        mapper = Mapper()
        
        #Reading patterns file
        lookupList = mapper.get_lookup_list(pattern_filepath)
        
        if verbose == True:
            print("Models Loaded!")
            print("---------------------------------")
            print(f'Input Folder path = {inp_dir}')
            print(f'Output Folder path = {output_dir_path}')
            print(f'Patterns File path = {pattern_filepath}')
            # print("Pattern List:\n", lookupList)
            print("---------------------------------")
        
        print("Splitting Started...")
        for name in tqdm(inFileList):
            input_filepath = os.path.join(inp_dir_path, name)
            if verbose == True:
                print("File Under processing: ", name)
                print("Path: ", input_filepath)
                
            #splitting audio file and saving results
            splitter.split_audio_file(input_filepath, output_dir_path, lookupList, verbose)

        print("Splitting Done!")
    
    except Exception as e :
        print("Error: 101!", e)
    
if __name__ == "__main__":
    main()