import cv2
import numpy as np
import easyocr
import os
import time
import requests
from io import BytesIO
from urllib.parse import urlparse

# Initialize OCR reader once
reader = easyocr.Reader(['en'])

def get_grade_ocr(source, debug_dir="debug_crops", scale=0.5):
    start_time = time.time()

    # Load image
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(source)

    if img is None:
        raise ValueError("Could not load image from source:", source)

    # Resize
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    h, w, _ = img.shape

    # Crop top portion
    top_crop = img[0:int(h*0.3), :]
    gray = cv2.cvtColor(top_crop, cv2.COLOR_BGR2GRAY)

    # Threshold for white background / black text
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area descending - PSA label is usually the largest contour
    contours_with_area = [(cnt, *cv2.boundingRect(cnt)) for cnt in contours]  # cnt, x, y, w, h
    contours_sorted = sorted(contours_with_area, key=lambda x: x[1]*x[3], reverse=True)

    # Ensure debug directory exists
    os.makedirs(debug_dir, exist_ok=True)

    crop_index = 0
    for cnt, x, y, w_rect, h_rect in contours_sorted:
        # Skip tiny regions
        if w_rect*h_rect < 500 or w_rect < 20 or h_rect < 10:
            continue

        aspect_ratio = w_rect / h_rect
        if aspect_ratio < 2.5 or aspect_ratio > 5:
            continue

        # Crop grayscale region
        grade_crop = gray[y:y+h_rect, x:x+w_rect]

        # Save for debugging
        crop_path = f"{debug_dir}/crop_{crop_index}.png"
        cv2.imwrite(crop_path, grade_crop)
        crop_index += 1

        # Focus on likely grade area
        h_crop, w_crop = grade_crop.shape
        focused_crop = grade_crop[int(h_crop*0.3):, int(w_crop*0.7):]

        # OCR
        result = reader.readtext(focused_crop)
        for _, text, prob in result:
            text_upper = text.upper()
            grade = None
            if 'MINT' in text_upper:
                grade = 9
            elif 'GEM MT' in text_upper:
                grade = 10

            if grade:
                print(f"Detected PSA grade: {grade} (confidence {prob:.2f})")
                print("Time:", time.time() - start_time)
                return grade

    print("No grade detected.")
    print("Time:", time.time() - start_time)
    return None

# Example usage
grade = process_card_image("card.jpg")  # local file
# grade = process_card_image("https://example.com/card.jpg")  # URL
