FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the necessary files for installation
COPY ./setup.py /app/setup.py
COPY ./requirements.txt /app/requirements.txt

# Copy the application code
COPY ./bot/bot.py /app/bot/bot.py
COPY ./bot/main.py /app/bot/main.py
COPY ./bot/__init__.py /app/bot/__init__.py
COPY ./model/__init__.py /app/model/__init__.py
COPY ./model/inference.py /app/model/inference.py
COPY ./model/input_preprocessing.py /app/model/input_preprocessing.py
COPY .env /app/.env

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install -e .

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Command to run the bot
CMD ["python", "bot/main.py"]
