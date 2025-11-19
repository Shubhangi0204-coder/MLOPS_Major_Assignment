# test.py
import joblib
from sklearn.metrics import accuracy_score

def main():
    obj = joblib.load("artifacts/savedmodel.pth")
    clf = obj["model"]
    X_test = obj["X_test"]
    y_test = obj["y_test"]

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()


