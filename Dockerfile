FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboards/app.py", "--server.address=0.0.0.0", "--server.port=8501"]