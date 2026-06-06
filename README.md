# Credit Card Default Prediction Service

Сервис машинного обучения для прогнозирования дефолта по кредитным картам.
Разработан в рамках учебного проекта по дисциплине «Внедрение моделей машинного обучения».

## Описание

Сервис принимает данные о клиенте банка и возвращает прогноз — допустит ли клиент
дефолт в следующем месяце. Реализованы две версии модели для A/B-тестирования:
- **v1** — Logistic Regression (базовая линейная модель)
- **v2** — Random Forest (ансамблевая модель)

Датасет: [Default of Credit Card Clients — UCI ML Repository](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

---

## Структура репозитория

```
credit-card-ml-deployment/
├── app/
│   ├── __init__.py
│   ├── api.py              # Flask-приложение, эндпоинты
│   └── model_handler.py    # Загрузка модели и инференс
├── models/
│   ├── train_model.py      # Скрипт обучения моделей
│   ├── model_v1.pkl        # Обученная модель v1
│   └── model_v2.pkl        # Обученная модель v2
├── tests/
│   └── test_api.py         # Тесты API
├── data/
│   └── UCI_Credit_Card.csv # Датасет
├── notebooks/              # Jupyter-ноутбуки для EDA
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── requirements.txt
├── ARCHITECTURE.md         # Описание архитектуры
├── ab_test_plan.md         # План A/B-тестирования
└── README.md
```

---

## Запуск локально

### 1. Клонировать репозиторий
```bash
git clone https://github.com/<your-username>/credit-card-ml-deployment.git
cd credit-card-ml-deployment
```

### 2. Создать виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Установить зависимости
```bash
pip install -r requirements.txt
```

### 4. Обучить модели
```bash
python models/train_model.py
```

### 5. Запустить сервис
```bash
python app/api.py
```

Сервис будет доступен на `http://localhost:5000`

---

## Запуск в Docker

### Собрать и запустить контейнер
```bash
docker build -t credit-card-ml .
docker run -p 5000:5000 credit-card-ml
```

### Запустить через Docker Compose (с nginx)
```bash
docker-compose up --build
```

### Скачать готовый образ с Docker Hub
```bash
docker pull niknik0900/credit-card-ml:latest
docker run -p 5000:5000 niknik0900/credit-card-ml:latest
```

Docker Hub: `https://hub.docker.com/r/niknik0900/credit-card-ml`

---

## API эндпоинты

### GET /health
Проверка работоспособности сервиса.

```bash
curl http://localhost:5000/health
```

Ответ:
```json
{"status": "healthy", "timestamp": "2025-01-01T12:00:00+00:00"}
```

---

### POST /predict
Прогноз дефолта. Параметр `?model=v1` или `?model=v2` (по умолчанию `v1`).

```bash
curl -X POST http://localhost:5000/predict?model=v1 \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
    "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689,
    "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
  }'
```

Ответ:
```json
{
  "prediction": 1,
  "probability": 0.7755,
  "model_version": "v1",
  "model_type": "LogisticRegression"
}
```

| Поле | Тип | Описание |
|---|---|---|
| `prediction` | int | 0 — нет дефолта, 1 — дефолт |
| `probability` | float | Вероятность дефолта (0.0–1.0) |
| `model_version` | string | Версия модели (v1 или v2) |
| `model_type` | string | Тип алгоритма |

---

### POST /predict/ab
A/B-тест: запрос случайно обрабатывается моделью v1 или v2.

```bash
curl -X POST http://localhost:5000/predict/ab \
  -H "Content-Type: application/json" \
  -d '{ ...те же признаки... }'
```

Ответ дополнительно содержит поле `ab_group`:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "model_version": "v2",
  "model_type": "RandomForestClassifier",
  "ab_group": "treatment"
}
```

---

### GET /features
Список признаков, которые принимает модель.

```bash
curl http://localhost:5000/features?model=v1
```

---

## Формат входных данных

| Признак | Тип | Описание |
|---|---|---|
| LIMIT_BAL | int | Кредитный лимит (NT dollar) |
| SEX | int | Пол: 1=мужской, 2=женский |
| EDUCATION | int | Образование: 1=аспирантура, 2=университет, 3=школа, 4=другое |
| MARRIAGE | int | Семейное положение: 1=женат, 2=холост, 3=другое |
| AGE | int | Возраст |
| PAY_0 | int | Статус оплаты в сентябре (-1=вовремя, 1..9=просрочка в месяцах) |
| PAY_2..PAY_6 | int | Статус оплаты с августа по апрель |
| BILL_AMT1..6 | int | Сумма счёта по месяцам (NT dollar) |
| PAY_AMT1..6 | int | Сумма фактической оплаты по месяцам (NT dollar) |

---

## Запуск тестов

```bash
python -m pytest tests/test_api.py -v
```
