FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY ./setup.py /app/setup.py
COPY ./requirements.txt /app/requirements.txt

COPY ./bot/bot.py /app/bot/bot.py
COPY ./bot/main.py /app/bot/main.py
COPY ./bot/__init__.py /app/bot/__init__.py
COPY ./model/__init__.py /app/model/__init__.py
COPY ./model/inference.py /app/model/inference.py
COPY ./model/input_preprocessing.py /app/model/input_preprocessing.py
COPY .env /app/.env

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install -e .

ENV PYTHONPATH=/app

CMD ["python", "bot/main.py"]
