FROM python:3.12-slim

WORKDIR /app

COPY hw2/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY hw2 /app/hw2

WORKDIR /app/hw2

CMD ["python", "-m", "src.models.train"]

