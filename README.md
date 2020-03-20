# UniverseEncoding

This file contains the information about the code
## universalEncodingForGDrive
Python code for Universal Encoding for accessing zipped files on Google Drive


To access the google drive, you need to turn on the Drive API. Currently I access the CLIENT CONFIGURATION file using https://developers.google.com/drive/api/v3/quickstart/python?authuser=0 url.


By clicking on "Enable Drive API" button, a json file is downloaded, you have to rename the file as "credentials.json".

`pip install pydrive`

### Accessing google drive using pydrive


`from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
drive = GoogleDrive(gauth)`

This code open your google account authentication in a new window.

### Accesssing the files in your google drive


### View all folders and file in your Google Drive

`fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file in fileList:
  print('Title: %s, ID: %s' % (file['title'], file['id']))
  # Get the folder ID that you want
  if(file['title'] == "To Share"):
      fileID = file['id']`
      
