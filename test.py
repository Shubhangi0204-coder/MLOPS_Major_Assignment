
import sys
import os

path = "artifacts/savedmodel.pth"
if not os.path.exists(path):
    print("ERROR: file not found:", path)
    sys.exit(1)


    try:
        import joblib
        obj = joblib.load(path)
        print("Loaded with joblib.load(), type:", type(obj))
    except Exception as e_joblib:
        print("joblib.load also failed:", e_joblib)
        # show a few bytes of file to help debugging
        with open(path, "rb") as f:
            head = f.read(128)
        print("first 128 bytes:", head[:128])
        raise
