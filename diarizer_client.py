import requests
import json

# Define the URL of the server-side API endpoint
url = "http://localhost:8000/uploadfile/"

# Define the path to the audio file to be processed
file_name='/home/idrak/Desktop/idrak_work/WhisperDiarizer/example_audios/clear.wav'

def send_file(url: str, file_path: str):
    """
    Sends an audio file to a server-side API endpoint for speaker diarization.

    Args:
        url (str): The URL of the server-side API endpoint.
        file_path (str): The path to the audio file to be processed.

    Returns:
        dict: The server-side response in JSON format.
    """
    
    # Define the headers for the POST request
    headers = {
        'accept': 'application/json',
    }
    
    # Open the audio file in binary mode
    with open(file_path, 'rb') as f:
        # Prepare the file data for the POST request
        files = {'file': f}
        # Send the POST request to the server-side API endpoint
        response = requests.post(url, headers=headers, files=files)
        # Convert the server-side response to JSON format
        json_response = response.json()
        print(json_response)
    
    # Return the server-side response
    return json_response


# Call the function to send the file and receive the response
recv_response = send_file(url, file_path=file_name)
# Print the received response and its type (dict)
print(recv_response, type(recv_response))
