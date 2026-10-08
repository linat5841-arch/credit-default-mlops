
# Использую официальный облегчённый образ Python 3.10
FROM python:3.10-slim

# Задаю рабочую директорию внутри контейнера
WORKDIR /app

# Отключаю создание файлов .pyc и буферизацию вывода
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Копирую список зависимостей
COPY requirements.txt .

# Устанавливаю необходимые библиотеки Python
RUN pip install --no-cache-dir -r requirements.txt

# Копирую исходный код приложения
COPY src/ ./src/

# Копирую обученную модель
COPY models/credit_default_model.joblib ./models/credit_default_model.joblib

# Документирую порт приложения
EXPOSE 8000

# Запускаю FastAPI через Uvicorn
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
