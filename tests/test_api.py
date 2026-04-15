from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_setosa():
    r = client.post("/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2
    })
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "setosa"
    assert data["confidence"] > 0.8

def test_predict_invalid_input():
    # Pydantic doit rejeter une valeur négative
    r = client.post("/predict", json={
        "sepal_length": -1.0, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2
    })
    assert r.status_code == 422  # Unprocessable Entity

def test_batch_limit():
    obs = [{"sepal_length":5.1,"sepal_width":3.5,
             "petal_length":1.4,"petal_width":0.2}] * 101
    r = client.post("/predict/batch", json={"observations": obs})
    assert r.status_code == 400