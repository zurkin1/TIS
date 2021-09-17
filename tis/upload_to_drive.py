from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pathlib import Path
import sys
import os

g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)

i = 0
for file in os.listdir('./hpa_p/'):
    if i%100 == 0:
        print(f'\r{i}')
    #file1 = drive.CreateFile({'parents': [{'id': 'TIS'}]})
    #file = 'hpa.zip'
    file1 = drive.CreateFile({'title': file, 'parents': [{'id': '1QnHbaBNtNHPcfXq1pWUIijYLuNJ6LgeR'}]})
    file1.SetContentFile('./hpa_p/'+file)
    file1['title'] = file
    file1.Upload()
    i += 1
