import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class MoodClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        print("Training Random Forest Classifier...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test)
        print("Model Training Complete.")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
    def predict(self, features):
        if not self.is_trained:
            raise Exception("Model is not trained yet!")
        # Reshape if single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.model.predict(features)
        
    def save_model(self, path='mood_classifier.pkl'):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path='mood_classifier.pkl'):
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            print(f"Model loaded from {path}")
        else:
            print("Model file not found.")
