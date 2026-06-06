FROM python:3.12-slim

WORKDIR /app

# Сначала копируем зависимости отдельно — это кэшируется Docker'ом.
# Если изменится только код, зависимости не будут переустанавливаться заново.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и обученные модели
COPY app/ ./app/
COPY models/ ./models/

# Открываем порт 5000
EXPOSE 5000

# Запускаем Flask-сервис
CMD ["python", "app/api.py"]
