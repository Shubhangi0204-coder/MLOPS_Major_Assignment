
from flask import Flask, request, render_template
import joblib
from PIL import Image
import numpy as np
import io

app = Flask(__name__)


model = joblib.load("model.pkl")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize((64, 64))
    img_array = np.array(img).reshape(1, -1).astype("float32") / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image_bytes = file.read()
            processed_img = preprocess_image(image_bytes)
            prediction = int(model.predict(processed_img)[0])

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
