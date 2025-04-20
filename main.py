from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import face_recognition
import uuid
import os
import json
from typing import List
from scipy.spatial.distance import cosine
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify domains like ["http://localhost:8081"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metadata_path = "metadata.json"

# Load face metadata
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        face_db = json.load(f)
else:
    face_db = []

SIMILARITY_THRESHOLD = 0.4  # Lower = more strict matching


def find_matching_person_id(new_encoding: List[float], face_db: List[dict]) -> str:
    for face in face_db:
        if "encoding" in face and face["encoding"] is not None:
            dist = cosine(new_encoding, face["encoding"])
            if dist < SIMILARITY_THRESHOLD:
                return face["person_id"]
    return None


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_path = f"temp_{uuid.uuid4()}.jpg"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    new_faces = []

    for location, encoding in zip(face_locations, face_encodings):
        matched_id = find_matching_person_id(encoding.tolist(), face_db)
        person_id = matched_id if matched_id else str(uuid.uuid4())

        face_data = {
            "original_file": file.filename,
            "face_file": f"{file.filename}_face.jpg",
            "location": {
                "top": location[0],
                "right": location[1],
                "bottom": location[2],
                "left": location[3],
            },
            "encoding": encoding.tolist(),
            "name": "Unknown",
            "person_id": person_id,
        }

        face_db.append(face_data)
        new_faces.append(face_data)

    with open(metadata_path, "w") as f:
        json.dump(face_db, f, indent=2)

    os.remove(image_path)
    return {"detected_faces": new_faces}


@app.delete("/reset/")
def reset_metadata():
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        return JSONResponse(content={"status": "reset complete"})
    return JSONResponse(content={"status": "no data to reset"})


@app.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)):
    contents = await file.read()
    image = face_recognition.load_image_file(np.frombuffer(contents, np.uint8))

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return {
        "face_locations": face_locations,
        "face_encodings": [enc.tolist() for enc in face_encodings],
        "num_faces": len(face_encodings)
    }
