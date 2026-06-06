import joblib
import numpy as np
import pandas as pd
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Загрузка модели
_models = {}
def load_model(version='v1'):
    if version not in _models:
        model_path = os.path.join(MODELS_DIR, f'model_{version}.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        _models[version] = joblib.load(model_path)

    return _models[version]

# Принимает словарь с признаками клиента,возвращает предсказание и вероятность дефолта
def predict(features: dict, version='v1') -> dict:

    artifact = load_model(version)
    model = artifact['model']
    scaler = artifact['scaler']
    feature_names = artifact['features']
    #Собираем признаки в DataFrame с правильными именами колонок
    try:
        X = pd.DataFrame([{f: features[f] for f in feature_names}])
    except KeyError as e:
        raise ValueError(f"Отсутствует признак: {e}")

    # Масштабируем
    if scaler is not None:
        X = scaler.transform(X)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return {
        'prediction': prediction,
        'probability': round(probability, 4),
        'model_version': version,
        'model_type': artifact['model_type']
    }


def get_feature_names(version='v1') -> list:
    artifact = load_model(version)
    return artifact['features']
