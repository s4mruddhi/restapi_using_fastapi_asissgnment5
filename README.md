# Iris Classification REST API using FastAPI

## 1. Objective

To train a machine learning model on the Iris dataset and expose it as a
REST API using FastAPI. The API accepts flower measurements in JSON format
and returns the predicted species with a confidence score.

## 2. Technologies Used

- Python 3.10 or later
- FastAPI (web framework for the API)
- Uvicorn (server that runs FastAPI, installed with `fastapi[standard]`)
- Pydantic (input validation, comes with FastAPI)
- scikit-learn (Random Forest model, Iris dataset)
- NumPy (array handling)
- Joblib (saving and loading the model)
- Pytest and HTTPX (automated testing)

## 3. Project Structure

```
fastapi-ml-assignment/
├── artifacts/
│   └── iris_model.joblib    # saved model + metadata
├── tests/
│   └── test_api.py          # automated tests
├── main.py                  # FastAPI application
├── train_model.py           # trains and saves the model
├── requirements.txt         # dependencies
├── pytest.ini               # tells pytest where main.py is
└── README.md
```

## 4. Installation Steps

### Create and activate a virtual environment

Windows:
```
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:
```
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```

## 5. Train the Model

```
python train_model.py
```

Expected output:
```
Model saved at: artifacts/iris_model.joblib
Test accuracy: 0.9000
```

## 6. Run the API

```
fastapi dev main.py
```
or
```
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

## 7. API Endpoints

| Method | Endpoint | Description |
|--------|-------------|--------------------------------------|
| GET | `/` | Welcome message |
| GET | `/health` | Checks API and model status |
| GET | `/model-info` | Model type, version, features, accuracy |
| POST | `/predict` | Predicts the Iris species |

### Input rules for `/predict`

Each of `sepal_length`, `sepal_width`, `petal_length`, `petal_width` must be
a number greater than 0 and less than or equal to 20.

### Example request

```
POST /predict
Content-Type: application/json

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Example response

```
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_version": "1.0.0"
}
```

### Example using cURL

```
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### HTTP status codes

| Code | Meaning |
|------|--------------------------------------|
| 200 | Request processed successfully |
| 404 | Endpoint not found |
| 422 | Input validation failed |
| 500 | Model could not generate a prediction |

## 8. Swagger Documentation

With the server running, open `http://127.0.0.1:8000/docs`.

1. Click `POST /predict`
2. Click **Try it out**
3. Enter the JSON input
4. Click **Execute**
5. Check the response body and status code

Test inputs to try:
- Valid input (above) -> 200
- `"sepal_length": -5` -> 422 (negative value)
- Remove `petal_width` -> 422 (missing field)
- `"sepal_length": "abc"` -> 422 (wrong data type)

Other documentation URLs: `/redoc` and `/openapi.json`.

## 9. Running the Tests

Run from the project root folder:

```
pytest -v
```

### Expected result

```
tests/test_api.py::test_root_endpoint PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_model_info_endpoint PASSED
tests/test_api.py::test_valid_prediction PASSED
tests/test_api.py::test_negative_feature_value PASSED
tests/test_api.py::test_missing_feature PASSED
tests/test_api.py::test_incorrect_data_type PASSED
tests/test_api.py::test_unknown_endpoint PASSED

8 passed
```

## 10. How the System Works

1. `train_model.py` trains a Random Forest on the Iris dataset and saves the
   model with its metadata into `artifacts/iris_model.joblib`.
2. When the FastAPI app starts, it loads this file once and keeps it in
   `app.state`.
3. A client sends flower measurements as JSON to `POST /predict`.
4. Pydantic validates the input. Invalid input is rejected with status 422.
5. The valid values are passed to the model, which predicts the species and
   class probabilities.
6. FastAPI returns the prediction, confidence, probabilities and model
   version as JSON.
