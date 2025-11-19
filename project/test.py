# Latest files — ready-to-run copies

Below are the latest copies of the files I recommended earlier. Paste these into your repo (matching paths shown) and run your CI. I included a minimal training placeholder so the upload/download flow is testable.

---

## File: .github/workflows/ci.yml

```yaml
name: CI - train & test (artifact)

on:
  push:
    branches:
      - dev

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run training and save model
        run: |
          mkdir -p artifacts
          # REPLACE the next line with your real training command that writes artifacts/savedmodel.pth
          # For testing we create a small binary file to simulate a model.
          python scripts/train.py --out artifacts/savedmodel.pth || echo "train script failed or not present"
          echo "After train: ls artifacts"
          ls -la artifacts
          echo "head of savedmodel.pth (hex)"
          head -c 256 artifacts/savedmodel.pth | hexdump -C || true

      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: savedmodel
          path: artifacts/savedmodel.pth

  test:
    runs-on: ubuntu-latest
    needs: train
    steps:
      - uses: actions/checkout@v4

      - name: Download model artifact
        uses: actions/download-artifact@v3
        with:
          name: savedmodel
          path: artifacts

      - name: Confirm artifact contents after download
        run: |
          echo "ls artifacts:"
          ls -la artifacts || true
          echo "file type:"
          file artifacts/savedmodel.pth || true
          echo "head (hex):"
          head -c 256 artifacts/savedmodel.pth | hexdump -C || true
          echo "first lines:"
          sed -n '1,40p' artifacts/savedmodel.pth || true

      - name: Install python deps including torch
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true
          pip install torch torchvision || true

      - name: Run tests
        run: python test.py
```

---

## File: test.py

```python
# test.py
import os, sys
path = "artifacts/savedmodel.pth"
print("Looking for:", path)
if not os.path.exists(path):
    print("ERROR: file not found at", path)
    sys.exit(10)

size = os.path.getsize(path)
print("File size (bytes):", size)
with open(path, "rb") as f:
    head = f.read(512)

print("First 200 bytes (text, errors replaced):")
print(head[:200].decode(errors="replace"))

# quick heuristic for workflow/yaml
if b"name:" in head[:256] and b"jobs:" in head[:256]:
    print("Likely a YAML/workflow file — not a model. Exiting.")
    sys.exit(11)

# Try loading with torch if available
try:
    import torch
    print("torch available, version:", torch.__version__)
    try:
        ckpt = torch.load(path, map_location="cpu")
        print("torch.load success. type:", type(ckpt))
        if isinstance(ckpt, dict):
            print("keys:", list(ckpt.keys())[:20])
        sys.exit(0)
    except Exception as e:
        print("torch.load failed:", repr(e))
except ModuleNotFoundError:
    print("torch not installed. Skipping torch.load.")

# Fallback to joblib
try:
    import joblib
    obj = joblib.load(path)
    print("joblib.load success. type:", type(obj))
    sys.exit(0)
except Exception as e:
    print("joblib.load failed:", repr(e))

print("All loaders failed. First 512 bytes printed above to help debugging.")
sys.exit(2)
```

---

## File: scripts/train.py (placeholder — replace with your real training)

```python
# scripts/train.py
# Minimal placeholder that writes a small binary file to simulate a model artifact.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
args = parser.parse_args()

content = b"\x00\x01\x02FAKE_PYTORCH_BINARY_MODEL\x03\x04"
with open(args.out, 'wb') as f:
    f.write(content)
print('Wrote dummy model to', args.out)
```

---

## File: requirements.txt

```
# Put your Python deps here. Example:
joblib
# Add other deps you need for your project; torch is installed in CI separately.
```

---

## Optional helper: test_load_torch.py (quick local check)

```python
# test_load_torch.py
import torch
path = 'artifacts/savedmodel.pth'
ckpt = torch.load(path, map_location='cpu')
print('Loaded type:', type(ckpt))
```

---

