# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def main():
    data = fetch_olivetti_faces()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )  

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump({
        "model": clf,
        "X_test": X_test,
        "y_test": y_test
    }, "model/savedmodel.pth")
    print("Model trained and saved to model/savedmodel.pth")

if __name__ == "__main__":
    main()
