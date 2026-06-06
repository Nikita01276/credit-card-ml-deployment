import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'UCI_Credit_Card.csv')
MODELS_DIR = os.path.dirname(__file__)


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['ID']) # Убираем ID - он не несёт информации для модели
    return df

def prepare_features(df):
    X = df.drop(columns=['default.payment.next.month'])
    y = df['default.payment.next.month']
    return X, y

# Логистическая регрессия
def train_v1(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'# учитываем дисбаланс классов
    )
    model.fit(X_scaled, y_train)

    #Сохраняем скейлер вместе с моделью в словаре
    artifact = {
        'model': model,
        'scaler': scaler,
        'features': list(X_train.columns),
        'version': 'v1',
        'model_type': 'LogisticRegression'
    }
    return artifact

# Случайный лес
def train_v2(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    artifact = {
        'model': model,
        'scaler': None,
        'features': list(X_train.columns),
        'version': 'v2',
        'model_type': 'RandomForestClassifier'
    }
    return artifact


def evaluate_model(artifact, X_test, y_test, version):
    model = artifact['model']
    scaler = artifact['scaler']

    if scaler is not None:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test
    y_pred = model.predict(X_input)
    f1 = f1_score(y_test, y_pred)


    print(f"Результаты модели {version}:")
    print(classification_report(y_test, y_pred, target_names=['Нет дефолта', 'Дефолт']))
    print(f"F1-score (дефолт): {f1:.4f}")
    return f1


def main():
    df = load_data(DATA_PATH)
    print(f"Датасет: {df.shape[0]} строк, {df.shape[1]} колонок")
    X, y = prepare_features(df)
    print(f"Распределение классов:\n{y.value_counts()}")

    # Разбивка данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nОбучающая выборка: {X_train.shape[0]} строк")
    print(f"Тестовая выборка: {X_test.shape[0]} строк")

    # Обучаем v1
    artifact_v1 = train_v1(X_train, y_train)
    evaluate_model(artifact_v1, X_test, y_test, 'v1')

    # Обучаем v2
    artifact_v2 = train_v2(X_train, y_train)
    evaluate_model(artifact_v2, X_test, y_test, 'v2')

    # Сохраняем модели
    path_v1 = os.path.join(MODELS_DIR, 'model_v1.pkl')
    path_v2 = os.path.join(MODELS_DIR, 'model_v2.pkl')

    joblib.dump(artifact_v1, path_v1)
    joblib.dump(artifact_v2, path_v2)
    
    
    
    
if __name__ == '__main__':
    main()
