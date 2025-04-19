
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import face_recognition
import shutil
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)):
    temp_filename = f"temp_{str(uuid.uuid4())}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = face_recognition.load_image_file(temp_filename)
    face_locations = face_recognition.face_locations(image)

    os.remove(temp_filename)

    return JSONResponse({
        "num_faces": len(face_locations),
        "face_locations": face_locations
    })
