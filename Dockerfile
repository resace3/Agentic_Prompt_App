ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

WORKDIR /app

COPY requirements.txt .
RUN if command -v python3 >/dev/null 2>&1; then PYTHON_BIN=python3; else PYTHON_BIN=python; fi; \
    if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then \
        apk add --no-cache python3 py3-pip; \
        PYTHON_BIN=python3; \
    fi; \
    "$PYTHON_BIN" -m pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY app.py .
COPY templates ./templates
COPY static ./static
COPY run.sh .
RUN chmod +x /app/run.sh

EXPOSE 5000
CMD ["/app/run.sh"]
