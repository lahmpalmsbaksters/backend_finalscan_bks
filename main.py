from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/process-images/")
async def process_images(background_image: UploadFile = File(...), current_frame: UploadFile = File(...)):
    # Read images directly from upload
    image_data1 = np.frombuffer(await background_image.read(), np.uint8)
    image_data2 = np.frombuffer(await current_frame.read(), np.uint8)
    background_image = cv2.imdecode(image_data1, cv2.IMREAD_COLOR)
    current_frame = cv2.imdecode(image_data2, cv2.IMREAD_COLOR)

    if background_image is None or current_frame is None:
        return JSONResponse(content={"message": "Failed to load one or both images"}, status_code=400)

    # Process images as per your background subtraction logic
    background_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    abs_diff = cv2.absdiff(background_gray, current_frame_gray)
    _, thresholded = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = current_frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Encode and return the images as base64 strings
    def encode_image_to_base64(cv_image):
        _, buffer = cv2.imencode('.png', cv_image)
        return base64.b64encode(buffer).decode('utf-8')

    base64_background = encode_image_to_base64(background_image)
    base64_current = encode_image_to_base64(current_frame)
    base64_result = encode_image_to_base64(result)

    return {
        "base64_background": base64_background,
        "base64_current": base64_current,
        "base64_processed": base64_result
    }


@app.get("/")
def read_root():
    return {"Hello": "World"}
