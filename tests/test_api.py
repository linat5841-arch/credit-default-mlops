from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app

# Создаю тестовый клиент FastAPI
client = TestClient(app)


@pytest.fixture
def valid_application():
    """Подготавливаю корректные данные клиента для тестирования."""

    return {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }


@pytest.fixture
def mock_model():
    """Подменяю обученную модель тестовым объектом."""

    # Создаю модель с заранее известными результатами
    model = MagicMock()
    model.predict.return_value = [1]
    model.predict_proba.return_value = np.array([[0.17, 0.83]])

    # Подменяю получение модели без загрузки файла
    with patch("src.api.get_model", return_value=model):
        yield model


def test_root_endpoint():
    """Проверяю доступность корневого endpoint."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "Credit Default Prediction API is running"}


def test_predict_success(valid_application, mock_model):
    """Проверяю успешное получение прогноза дефолта."""

    # Отправляю корректные данные клиента
    response = client.post("/predict", json=valid_application)

    # Вывожу диагностическую информацию при ошибке
    if response.status_code != 200:
        print("API status:", response.status_code)
        print("API response:", response.text)

        # Проверяю исходное исключение при наличии
        if hasattr(response, "extensions"):
            print("Response extensions:", response.extensions)

    # Проверяю HTTP-статус
    assert response.status_code == 200

    # Проверяю структуру и значения ответа
    result = response.json()

    assert result["prediction"] == 1
    assert result["default_probability"] == pytest.approx(0.83)

    # Проверяю, что модель действительно использовалась
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()


def test_probability_range(valid_application, mock_model):
    """Проверяю диапазон вероятности дефолта."""

    response = client.post("/predict", json=valid_application)

    assert response.status_code == 200

    probability = response.json()["default_probability"]

    assert 0 <= probability <= 1


def test_invalid_input_rejected(valid_application):
    """Проверяю отклонение некорректных данных клиента."""

    # Создаю заведомо некорректный кредитный лимит
    invalid_application = valid_application.copy()
    invalid_application["LIMIT_BAL"] = -1000

    # Отправляю некорректный запрос
    response = client.post("/predict", json=invalid_application)

    # Проверяю, что Pydantic отклоняет запрос
    assert response.status_code == 422
