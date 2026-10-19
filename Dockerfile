FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY reports/ reports/

EXPOSE 5000
CMD ["python", "api/app.py"]
