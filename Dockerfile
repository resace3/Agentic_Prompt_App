ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates ./templates
COPY static ./static
COPY run.sh .
RUN chmod +x /app/run.sh

EXPOSE 5000
CMD ["/app/run.sh"]
