FROM python:3.9-slim
LABEL authors="nirkon"

WORKDIR /app
COPY ratings_server.py /app
COPY inference_models/. /app/inference_models
COPY models/Model_full_data/2/. /app/models/Model_full_data/2/
COPY models/random_forest_review_model.joblib /app/models/random_forest_review_model.joblib
COPY templates/. /app/templates
COPY static/. /app/static
COPY requirements_docker.txt /app

# Install build dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements_docker.txt && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /root/.cache/pip/*

RUN rm -rf /tmp/* /var/tmp/*

EXPOSE 8081

CMD ["python", "ratings_server.py"]