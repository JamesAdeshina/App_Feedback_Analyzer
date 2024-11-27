from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_model(X, y):
    """Train a logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

if __name__ == "__main__":
    import pandas as pd
    from scipy.sparse import csr_matrix
    from feature_engineer import generate_tfidf_features

    data = pd.read_csv("../data/processed_reviews.csv")
    tfidf_matrix, _ = generate_tfidf_features(data, 'review')
    model = train_model(tfidf_matrix, data['star'])
    joblib.dump(model, "../models/saved_model.pkl")
