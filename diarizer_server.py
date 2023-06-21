from fastapi import FastAPI, UploadFile, File
from pyannote_diarizer import Py_Diarizer
import os

# Initializing the Whisper Model and Authorization token from Hugging Face
whisper_model, auth_token = "base.en", "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"

# Specifying the path of the output directory
outfolder_path = "OUTPUT"

# Initialize the Py_Diarizer object with the authentication token
diarizer = Py_Diarizer(auth_token)

# Initialize the FastAPI application
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    """
    Uploads an audio file and performs speaker diarization.
    
    Args:
        file (UploadFile): Audio file to be uploaded and processed.
        
    Returns:
        dict: A dictionary containing the status of the operation 
              and either the speaker-tagged segments if successful, 
              or an error message in case of failure.
    """
    
    # Define the location for the uploaded files
    file_location = f"uploads/{file.filename}"
    
    # Open and save the uploaded file
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    # Perform speaker diarization
    tagged_segments = diarizer.seprate_speakers(file_location, outfolder_path)
    
    # Prepare and return the output
    if tagged_segments:
        output = {
            'status': True,
            'msg': tagged_segments
        }
    else:
        output = {
            'status': False,
            'msg': {
                'error': 'Something bad happens'
            }
        }
    
    return output
