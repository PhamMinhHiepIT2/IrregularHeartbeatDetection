FROM python:3.8

WORKDIR /opt

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

COPY scripts/ ./scripts
COPY model/ ./model

RUN apt update \
    && apt install -y python3-opencv \
    && rm -rf /var/lib/apt/lists/*

CMD [ "python", "scripts/app.py" ]