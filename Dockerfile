# syntax=docker/dockerfile:1.2
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY challenge/ ./challenge/
COPY data/data.csv ./data/data.csv

RUN python -c "\
import pandas as pd; \
from challenge.model import DelayModel; \
m = DelayModel(); \
features, target = m.preprocess(pd.read_csv('data/data.csv', low_memory=False), target_column='delay'); \
m.fit(features, target)"

EXPOSE 8080

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
