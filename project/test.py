import os, sys

path = "artifacts/savedmodel.pth"
if not os.path.exists(path):
    print("ERROR: model file not found at", path)
    sys.exit(1)

# quick sanity inspect
with open(path, "rb") as f:
    head = f.read(256)
try:
    # check if file looks like text (workflow YAML)
    if head.startswith(b'name:') or b'on:' in head[:80] or head.strip().startswith(b'---'):
        print("ERROR: file looks like a text/workflow file, not a model. First bytes:")
        print(head.decode(errors='replace')[:400])
        sys.exit(2)
except Exception:
    pass

# try torch first
try:
    import torch
    ckpt = torch.load(path, map_location="cpu")
    print("Loaded with torch.load(). type:", type(ckpt))
    sys.exit(0)
except ModuleNotFoundError as e:
    print("torch not installed:", e)
except Exception as e:
    print("torch.load raised:", repr(e))

# fallback to joblib
try:
    import joblib
    obj = joblib.load(path)
    print("Loaded with joblib. type:", type(obj))
except Exception as e:
    print("joblib.load failed:", repr(e))
    print("first 256 bytes (raw):", head)
    raise

