FROM python:3.9-slim-buster
WORKDIR /app
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install feast[redis] --index-url https://pypi-token:glpat-41XJLFL_fayjoWH2Tjwk@gitlab.com/api/v4/projects/35900603/packages/pypi/simple
ENV PYTHONPATH "/app"
ENV FEAST_S3_ENDPOINT_URL "http://34.101.155.219:9000"
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
COPY . /app
