FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV WANDB_MODE=offline
COPY . .
ENTRYPOINT ["python", "inference.py"]
