from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from app.main import video_inference
from app.pipeline import EmotionPerceptionTool
import numpy as np
import uuid
import cv2
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil
import os

app = FastAPI()

def process_image_request(contents):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    #https://www.codegrepper.com/code-examples/python/fastapi+upload+video+file+example
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

def create_validate_tmp_folder():
    tmp_path = os.getcwd() + "/tmp/"
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    return

emotion_perception = EmotionPerceptionTool(model_path='./app/IOEPT_v0.1.pt')

@app.get("/")
def read_root():
    return {"Humath": "Instance Of Emotion Perception Tool"}

@app.post("/image_inference/")
async def process_image_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    file.filename = f"{uuid.uuid4()}.{suffix}]"
    contents = await file.read()  # <-- Important!
    img = process_image_request(contents)
    result = emotion_perception(img)

    emotions = result['data']['emotions']
    face_count = result['detector']['face_count']
    return {"filename": file.filename,"face_count": face_count, "emotions": emotions}

@app.post("/video_inference/")
async def process_video_file(upload_file: UploadFile) -> None:
    create_validate_tmp_folder()
    suffix = Path(upload_file.filename).suffix
    tmp_file = f"/tmp/temp_video" + suffix
    save_path = Path(os.getcwd() + tmp_file)
    save_upload_file(upload_file, save_path)
    print('file uploaded.')
    response_path = video_inference(str(save_path))
    return FileResponse(path=Path(response_path), filename="procesed.mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)