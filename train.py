# train_model.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib, pickle
import os

def main():
    print("Loading Olivetti dataset...")
    data = fetch_olivetti_faces()
    X = data.images
    y = data.target

    X_flat = X.reshape((X.shape[0], -1))


    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Training Decision Tree classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # save model
    joblib.dump(clf, "model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    main()
