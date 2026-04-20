import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "model/mask_detector.h5"
model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

LABELS = ["with_mask", "without_mask"]
COLORS = {"with_mask": (0, 255, 0), "without_mask": (0, 0, 255)}


def detect_and_predict(image_array: np.ndarray) -> dict:
    # Make a copy to draw on
    output_image = image_array.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram to improve detection in different lighting
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    results = []

    if len(faces) == 0:
        return {
            "faces_detected": 0,
            "results": [],
            "annotated_image": output_image
        }

    for (x, y, w, h) in faces:
        # Add padding around face
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_array.shape[1], x + w + pad)
        y2 = min(image_array.shape[0], y + h + pad)

        face_roi = image_array[y1:y2, x1:x2]

        if face_roi.size == 0:
            continue

        # Preprocess for CNN
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (100, 100))
        face_arr = img_to_array(face_resized) / 255.0
        face_arr = np.expand_dims(face_arr, axis=0)

        # Predict
        preds = model.predict(face_arr, verbose=0)[0]
        label_idx = np.argmax(preds)
        label = LABELS[label_idx]
        confidence = float(preds[label_idx]) * 100
        color = COLORS[label]

        # Draw rectangle
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)

        # Draw filled label background
        label_text = f"{label.replace('_', ' ')}: {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output_image, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(
            output_image, label_text,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 2
        )

        results.append({
            "label": label,
            "confidence": round(confidence, 2),
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        })

    return {
        "faces_detected": len(results),
        "results": results,
        "annotated_image": output_image
    }