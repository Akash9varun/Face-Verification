from email.mime import image
from typing import Optional
from fastapi import FastAPI
import numpy as np
import face_recognition
import cv2
import urllib.request
from pydantic import BaseModel

app = FastAPI()

class DataModel(BaseModel):
    url:str

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

@app.get('/')
def serverstatus():
    return {"status": "Running"}

@app.post("/text/")
def read_root(dm:DataModel):
    baseurl = dm.url
    image = url_to_image(baseurl)
    
    picture_of_me = face_recognition.load_image_file(image)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
    unknown_picture = face_recognition.load_image_file(image)
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
    if results[0] == True:
        return{"result": "Face Matched"}

    else:
        return{"result": "It's not a picture of me!"}


