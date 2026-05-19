from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import gc
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

# Limit concurrent inferences to prevent OOM under load
_semaphore = asyncio.Semaphore(2)

MAX_IMAGE_SIZE = (640, 640)
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large. Max 10MB.")

    async with _semaphore:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.draft("RGB", MAX_IMAGE_SIZE)
            image = image.convert("RGB")
            image.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)

            results = model(image, verbose=False)

            detections = []
            has_logo = False

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    conf = float(box.conf[0])

                    if label == "logo":
                        has_logo = True

                    detections.append({"label": label, "confidence": round(conf, 3)})
        finally:
            del image_bytes, image
            gc.collect()

    return {"hasLogo": has_logo, "detections": detections}
