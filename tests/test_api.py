import sys
import os
import json
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from api import app

# Тестовый набор признаков (первый клиент из датасета)
SAMPLE_CLIENT = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
    "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689,
    "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


def test_predict_v1(client):
    response = client.post(
        '/predict?model=v1',
        data=json.dumps(SAMPLE_CLIENT),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability' in data
    assert data['model_version'] == 'v1'
    assert data['prediction'] in [0, 1]
    assert 0.0 <= data['probability'] <= 1.0


def test_predict_v2(client):
    response = client.post(
        '/predict?model=v2',
        data=json.dumps(SAMPLE_CLIENT),
        content_type='application/json'
    )
    assert response.status_code ==200
    data = json.loads(response.data)
    assert data['model_version'] == 'v2'


def test_predict_ab(client):
    response = client.post(
        '/predict/ab',
        data=json.dumps(SAMPLE_CLIENT),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['ab_group'] in ['control', 'treatment']


def test_predict_invalid_model(client):
    response = client.post(
        '/predict?model=v99',
        data=json.dumps(SAMPLE_CLIENT),
        content_type='application/json'
    )
    assert response.status_code== 400


def test_predict_empty_body(client):
    response = client.post(
        '/predict',
        data='',
        content_type='application/json'
    )
    assert response.status_code == 400


def test_predict_missing_feature(client):
    bad_data = {"LIMIT_BAL": 20000}
    response = client.post(
        '/predict',
        data=json.dumps(bad_data),
        content_type='application/json'
    )
    assert response.status_code == 400
