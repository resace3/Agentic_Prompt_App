ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

WORKDIR /app

COPY requirements.txt .
RUN if command -v apk >/dev/null 2>&1; then \
        apk add --no-cache python3 py3-pip py3-matplotlib; \
        grep -viE '^matplotlib==' requirements.txt > /tmp/requirements-docker.txt; \
        python3 -m pip install --no-cache-dir --break-system-packages -r /tmp/requirements-docker.txt; \
    else \
        python -m pip install --no-cache-dir -r requirements.txt; \
    fi

COPY app.py .
COPY jitai_engine.py .
COPY config.yaml .
COPY templates ./templates
COPY static ./static
COPY run.sh .
RUN if [ ! -x /usr/bin/with-contenv ]; then \
        printf '%s\n' '#!/bin/sh' 'exec "$@"' > /usr/bin/with-contenv && chmod +x /usr/bin/with-contenv; \
    fi && \
    chmod +x /app/run.sh

EXPOSE 5000
CMD ["/app/run.sh"]
