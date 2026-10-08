from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prepare import clean_data, create_features

# Определяю корневую директорию проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Задаю путь к обученной модели
MODEL_PATH = PROJECT_ROOT / "models" / "credit_default_model.joblib"

# Создаю FastAPI-приложение
app = FastAPI(
    title="Credit Default Prediction API",
    description="API для прогнозирования вероятности дефолта клиента",
    version="1.0.0",
)


class CreditApplication(BaseModel):
    """Описываю входные признаки клиента для скоринговой модели."""

    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=21, le=100)

    PAY_0: int = Field(ge=-2, le=8)
    PAY_2: int = Field(ge=-2, le=8)
    PAY_3: int = Field(ge=-2, le=8)
    PAY_4: int = Field(ge=-2, le=8)
    PAY_5: int = Field(ge=-2, le=8)
    PAY_6: int = Field(ge=-2, le=8)

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Описываю формат ответа скорингового API."""

    prediction: int
    default_probability: float


def load_model():
    """Загружаю обученную модель из файла."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


# Откладываю загрузку модели до первого запроса
model = None


def get_model():
    """Получаю модель, загружая её при необходимости."""
    global model

    if model is None:
        model = load_model()

    return model


def create_model_input(application: CreditApplication) -> pd.DataFrame:
    """Подготавливаю входные данные для сохранённого ML pipeline."""

    # Преобразую входные данные клиента в DataFrame
    input_df = pd.DataFrame([application.model_dump()])

    # Применяю ту же очистку, что и при обучении
    input_df = clean_data(input_df)

    # Создаю дополнительные признаки по общей логике
    input_df = create_features(input_df)

    return input_df


@app.get("/")
def root() -> dict[str, str]:
    """Проверяю доступность API."""

    return {"status": "Credit Default Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    """Прогнозирую класс и вероятность дефолта клиента."""

    try:
        # Подготавливаю признаки клиента
        input_df = create_model_input(application)

        # Получаю обученную модель
        trained_model = get_model()

        # Получаю прогноз класса
        prediction = int(trained_model.predict(input_df)[0])

        # Получаю вероятность дефолта
        default_probability = float(trained_model.predict_proba(input_df)[0, 1])

        return PredictionResponse(
            prediction=prediction,
            default_probability=default_probability,
        )

    except Exception as error:
        # Возвращаю безопасное сообщение при ошибке прогнозирования
        raise HTTPException(
            status_code=500,
            detail="Не удалось выполнить прогноз.",
        ) from error
