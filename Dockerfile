FROM python:3.8

WORKDIR /app

COPY requirements/requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN pip install --upgrade pip 

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
