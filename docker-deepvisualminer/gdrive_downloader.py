# Small command line downloader for Google Drive URLs.
#
# Thanks to user http://stackoverflow.com/users/6941051/user115202
#   from http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive

from __future__ import print_function

import requests

import sys

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

        save_response_content(response, destination)    
    else:
        print("Unable to download. Cannot get confirm token")



def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None
    
    

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    total = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                total += len(chunk)
                sys.stdout.write("{:,} bytes downloaded\r".format(total))
                sys.stdout.flush()
                
    print("Download completed: {} [{:,} bytes]".format(destination, total))



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 gdrive_downloader.py FILE-ID DESTINATION-FILE-PATH\n" + \
            " where FILE_ID is the value of 'id' parameter in the file's shareable link")
            
        sys.exit(1)
        
    file_id = sys.argv[1]
    destination_file = sys.argv[2]
    download_file_from_google_drive(file_id, destination_file)
