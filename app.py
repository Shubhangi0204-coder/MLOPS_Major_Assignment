
from flask import Flask, request, render_template_string
import joblib
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import sys

app = Flask(__name__)
MODEL_PATH = Path("artifacts/savedmodel.pth")

if not MODEL_PATH.exists():
    
    print(f"Model not found at {MODEL_PATH}. Run train.py to create it.", file=sys.stderr)
    raise SystemExit(1)

obj = joblib.load(MODEL_PATH)
model = obj.get("model") if isinstance(obj, dict) else obj

HTML = """
<!doctype html>
<title>My Olivetti predictor</title>
<h1>Upload 64x64 grayscale face</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
<h2>Predicted class: {{ pred }}</h2>
{% endif %}
"""

def preprocess(img_bytes):
    img = Image.open(BytesIO(img_bytes)).convert("L").resize((64,64))
    arr = np.array(img).reshape(1, -1)
    return arr

@app.route("/", methods=["GET","POST"])
def index():
    pred = None
    if request.method == "POST":
        f = request.files.get("file")
        if f:
            arr = preprocess(f.read())
            pred = int(model.predict(arr)[0])
    return render_template_string(HTML, pred=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
