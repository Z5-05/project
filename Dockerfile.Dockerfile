FROM python:3.8.1-alpine

WORKDIR /app/src

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /app/requirements.txt

# install dependencies
RUN set -eux
RUN apk add --no-cache --virtual .build-deps build-base \
        libressl-dev libffi-dev gcc musl-dev python3-dev
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN rm -rf /root/.cache/pip

# copy project
COPY src /app/src
CMD uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8000
