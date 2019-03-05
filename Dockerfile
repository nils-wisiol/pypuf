FROM intelpython/intelpython3_full

RUN mkdir /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD python3
