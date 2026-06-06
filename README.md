# Credit Card Default Prediction Service
ML-сервис для прогнозирования дефолта по кредитным картам

## Модели
- **v1** — Logistic Regression (F1 = 0.46)
- **v2** — Random Forest (F1 = 0.54)

### Запуск локально

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python models/train_model.py
python app/api.py
```

## Запуск в Docker

```bash
docker pull niknik0900/credit-card-ml:latest
docker run -p 5000:5000 niknik0900/credit-card-ml:latest
```

### API эндпоинты

| Метод | Эндпоинт | Описание |
|---|---|---|
| GET |/health | Проверка сервиса |
| POST | /predict?model=v1 | Предсказание моделью v1 или v2 |
| POST |/predict/ab | A/B-тест (случайная модель) |
| GET | /features | Список признаков |

## Пример запроса

```bash
curl.exe -X POST "http://localhost:5000/predict?model=v1" \
  -H "Content-Type: application/json" \
  -d '@test_data.json'
```

## Структура проекта

```
credit-card-ml-deployment/
│
├── app/
│   ├── __init__.py         
│   ├── api.py              # Flask-приложение
│   └── model_handler.py    # Загрузка и использование модели
│
├── models/
│   ├── model_v1.pkl        # Модельv1
│   ├── model_v2.pkl        # Модельv2
│   └── train_model.py      # Скрипт обучения моделей
│
├── notebooks/
│   └── eda.ipynb           # Разведочный анализ данных
│
├── tests/
│   └── test_api.py         # Тесты API
│
├── .gitignore
├── ARCHITECTURE.md         # Описание архитектуры
├── Dockerfile              # Конфигурация Docker
├── README.md               # Основная документация
├── ab_test_plan.md         # План A/B-тестирования
├── docker-compose.yml      # Оркестрация сервисов
├── nginx.conf              # Конфигурация Nginx
└── requirements.txt        # Зависимости Python
```

Ссылка на Docker
https://hub.docker.com/r/niknik0900/credit-card-ml
