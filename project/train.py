# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np
import os

def main():
    # load dataset
    data = fetch_olivetti_faces()
    X = data.images  # shape (400, 64, 64)
    y = data.target  # labels 0-39

    # flatten images for classic sklearn classifier
    n_samples = X.shape[0]
    X_flat = X.reshape((n_samples, -1))

    # split 70% train / 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, train_size=0.7, random_state=42, stratify=y
    )

    # train DecisionTree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # save model and test set (so test.py can load test data)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({"model": clf, "X_test": X_test, "y_test": y_test}, "artifacts/savedmodel.pth")
    print("artifacts/savedmodel.pth")

if __name__ == "__main__":
    main()

