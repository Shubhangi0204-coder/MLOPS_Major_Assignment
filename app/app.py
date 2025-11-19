
import os
import io
import joblib
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np

app = Flask(__name__)
MODEL_PATH = os.path.join("artifacts", "savedmodel.pth")
data = joblib.load(MODEL_PATH)
clf = data["artifacts"]

# Olivetti faces images are 64x64 grayscale flattened
def preprocess_image(file_storage):
    img = Image.open(file_storage).convert("L").resize((64,64))
    arr = np.array(img).astype("float32").reshape(1, -1) / 255.0
    return arr

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        f = request.files["image"]
        arr = preprocess_image(f)
        pred = clf.predict(arr)[0]
        return render_template("index.html", result=f"Predicted class: {pred}")
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
