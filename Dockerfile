FROM python:3.9-slim

WORKDIR /app

EXPOSE 8271

COPY requirements.txt ./
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./app .

#CMD [ "python", "-u","main.py" ]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8271"]