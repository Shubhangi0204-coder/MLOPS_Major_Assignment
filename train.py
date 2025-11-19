# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def main():
    print("Loading Olivetti dataset...")
    data = fetch_olivetti_faces()
    X = data.images
    y = data.target

    # Flatten images (64x64 -> 4096)
    X_flat = X.reshape((X.shape[0], -1))

    # 70% train / 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, train_size=0.7, random_state=42, stratify=y
    )

    print("Training Decision Tree classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)

    # Save model and test split so test.py can evaluate it
    joblib.dump({"model": clf, "X_test": X_test, "y_test": y_test}, "artifacts/savedmodel.pth")
    print("Saved model to artifacts/savedmodel.pth")

if __name__ == "__main__":
    main()
