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

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def preprocess_and_straighten(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        straightened = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    else:
        straightened = image.copy()

    return straightened

def align_images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width, channels = image1.shape
    aligned_image2 = cv2.warpPerspective(image2, H, (width, height))
    return aligned_image2

def compare_images(background_image, current_frame):
    Y, X, _ = background_image.shape
    y, x, _ = current_frame.shape
    new_x = max(X, x)
    new_y = max(Y, y)

    background_image = cv2.resize(background_image, (new_x, new_y))
    current_frame = cv2.resize(current_frame, (new_x, new_y))

    background_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    abs_diff = cv2.absdiff(background_gray, current_frame_gray)
    _, thresholded = cv2.threshold(abs_diff, 70, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = current_frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result

@app.post("/process-images/")
async def process_images(background_image: UploadFile = File(...), current_frame: UploadFile = File(...)):
    # Read images directly from upload
    image_data1 = np.frombuffer(await background_image.read(), np.uint8)
    image_data2 = np.frombuffer(await current_frame.read(), np.uint8)
    background_image = cv2.imdecode(image_data1, cv2.IMREAD_COLOR)
    current_frame = cv2.imdecode(image_data2, cv2.IMREAD_COLOR)

    if background_image is None or current_frame is None:
        return JSONResponse(content={"message": "Failed to load one or both images"}, status_code=400)

    # Preprocess and straighten the images
    preprocessed_background = preprocess_and_straighten(background_image)
    preprocessed_current = preprocess_and_straighten(current_frame)

    # Align the straightened current image with the background image
    aligned_current = align_images(preprocessed_background, preprocessed_current)

    # Compare the aligned image with the background image
    comparison_result = compare_images(preprocessed_background, aligned_current)

    # Encode and return the images as base64 strings
    def encode_image_to_base64(cv_image):
        _, buffer = cv2.imencode('.png', cv_image)
        return base64.b64encode(buffer).decode('utf-8')

    base64_background = encode_image_to_base64(preprocessed_background)
    base64_current = encode_image_to_base64(aligned_current)
    base64_result = encode_image_to_base64(comparison_result)

    return {
        "base64_background": base64_background,
        "base64_current": base64_current,
        "base64_processed": base64_result
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}
