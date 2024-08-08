FROM python:3.12

WORKDIR /Makemytrip_Hotel_Recommendation

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY Makemytrip_Hotel_Recommendation.py .

COPY dataset.csv .

EXPOSE 8501

CMD ["streamlit", "run", "Makemytrip_Hotel_Recommendation.py"]

