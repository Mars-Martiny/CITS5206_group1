import numpy as np

class DummyModel:
    def fit(self, X, y):
        print("[INFO] Dummy training completed")

    def predict(self, X):
        return ["Energy"] * len(X)

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]] * len(X))
