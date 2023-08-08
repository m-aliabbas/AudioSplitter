# IdrakDiarizer

It is a speaker diarization application that separates the different speakers in an audio file. It uses FastWhisper for transcription, Pyannote Diarization
pipeline for speaker identification, and fastapi for request and respons. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the application, make sure you have the following dependencies installed:

- FastAPI
- pyannote
- requests
- uvicorn (for running the FastAPI application)

You can install these packages using pip:

```
pip install fastapi pyannote.core requests uvicorn faster_whisper
```

### Running the Application

To run the application, you first need to start the FastAPI server. To do this, navigate to the directory containing the diarizer_server.py file and run the following command:

```

uvicorn diarizer_server:app --host 192.168.100.100  --port 8080 --reload



```
This will start the FastAPI application on your localhost.

Then, you can use the diarizer_client.py script to send an audio file to the server for speaker diarization. Simply replace the file_name variable with the path to your audio file and run the script.

### Usage
Do send a post request to `uploadfile` route of server url. It will return a json response. For example ussage please see the `diarizer_client.py`
The application provides a FastAPI endpoint for uploading an audio file. Once the file is uploaded, the server performs speaker diarization and returns the tagged speaker segments if successful, or an error message if not.

```

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # ... rest of code ...
```

### Authors

    - Mohammad Musawir Baig
    - Mohammad Ali Abbas

License

This property of company (IdrakAi)

