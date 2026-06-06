import random
import logging
import json
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from model_handler import predict, get_feature_names

app = Flask(__name__)

# Настройка логирования вJSON-формате
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Логирует запрос и ответ в JSON-формате
def log_request(endpoint, request_data, response_data):
    log_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'endpoint': endpoint,
        'request': request_data,
        'response': response_data
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))



# GET /health
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    version = request.args.get('model', 'v1')
    if version not in ('v1', 'v2'):
        return jsonify({'error': 'Параметр model должен быть v1 или v2'}), 400

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Тело запроса пустое илине является JSON'}), 400

    try:
        result = predict(data, version=version)
        log_request('/predict', data, result)
        return jsonify(result), 200
    except ValueError as e:
        error = {'error': str(e)}
        log_request('/predict', data, error)
        return jsonify(error), 400


# A/B-тест:случайно направляет запрос на v1 или v2 
# Возвращает предсказание и информацию о том, какая модель
@app.route('/predict/ab', methods=['POST'])
def predict_ab():
    data =request.get_json()
    if not data:
        return jsonify({'error': 'Тело запроса пустое или не является JSON'}), 400

    # Случайное  распределение 50/50
    version = random.choice(['v1', 'v2'])
    try:
        result = predict(data, version=version)
        result['ab_group'] = 'control' if version == 'v1' else 'treatment'
        log_request('/predict/ab', data, result)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


# Вспомогательный эндпоинт - показывает список признаков
@app.route('/features', methods=['GET'])
def features():
    version = request.args.get('model', 'v1')
    try:
        names = get_feature_names(version)
        return jsonify({'model_version': version, 'features': names}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
