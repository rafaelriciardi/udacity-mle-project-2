# Unit tests for the model training proccess

import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

#Check the processing functions
def test_process_data(data):
    train, test = train_test_split(data, test_size=0.20)
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    assert len(data) == len(train) + len(test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

#Check if training is returning the correct object type
def test_training_step(X, y):
    trained_model = train_model(X, y)
    assert isinstance(trained_model, RandomForestClassifier)

#Check if saved model file is correct
def test_saved_model():
    # Load the model
    saved_model = joblib.load("../model/model.pkl")
    assert isinstance(saved_model, RandomForestClassifier)