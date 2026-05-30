FROM python:3.14
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app_backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]