FROM python:3.9.12-slim-buster

WORKDIR /app

COPY . .

RUN pip3 install -r ./requirements.txt

EXPOSE 8501 

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]