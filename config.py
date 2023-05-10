import torch

# git clone https://github.com/Idrak-Pak/whisper.git
WISPER_PATH = "/home/mmb/Desktop/AfterGrad/IdrakWork/whisper/WTranscriptor"

config = dict()
# -------------- General configs ------------#
config["samplerate"] = 16000
config["cuda_device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------- Diarization configs ------------#
config["num_speakers"] = 2 #@param {type:"integer"}
config["language"] = 'English' #@param ['any', 'English']
config["model_size"] = 'small.en' #@param ['tiny', 'base', 'small', 'medium', 'large']
# print("MODEL SIZE: ", config["model_size"])
if config["language"] == 'English' and config["model_size"] != 'large':
  config["model_name"] = config["model_size"] + '.en'
  
# -------------- Splitter configs ------------#
config["avatar"] = "Amy" #Becky, Ethan
if config["avatar"] == "Amy":
  config["patterns_path"] = "./Patterns/Amy_Patterns_Dec3rd.csv"
if config["avatar"] == "Becky":
  config["patterns_path"] = "./Patterns/Becky_Patterns-Nov30thFixed.csv"  
if config["avatar"] == "Ethan":
  config["patterns_path"] = "./Patterns/ethan_patterns_Mar_6th.csv"  