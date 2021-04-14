FROM gcr.io/datamechanics/spark-py-connectors:3.0

WORKDIR /opt/application

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/ src/
COPY main.py .

ENV PYSPARK_MAJOR_VERSION=3
