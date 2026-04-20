import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from detector import detect_and_predict

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "MaskGuard API is running 🎭"})


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    POST /api/detect
    Body: { "image": "<base64 encoded image>" }
    Returns: { "faces_detected": N, "results": [...], "image": "<annotated base64>" }
    """
    body = request.get_json()
    if not body or "image" not in body:
        return jsonify({"error": "image field required"}), 400

    try:
        # Decode base64 image from frontend
        img_data = base64.b64decode(body["image"])
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Run detection
        output = detect_and_predict(image)

        # Encode annotated image back to base64
        _, buffer = cv2.imencode(".jpg", output["annotated_image"])
        encoded = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "faces_detected": output["faces_detected"],
            "results": output["results"],
            "image": encoded
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)