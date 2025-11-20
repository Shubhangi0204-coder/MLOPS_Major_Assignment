import joblib
from sklearn.metrics import accuracy_score

print("Loading saved model...")
data = joblib.load("artifacts/savedmodel.pth")
model = data["model"]
X_test = data["X_test"]
y_test = data["y_test"]

print("Running predictions...")
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
print("Test accuracy:", acc)


