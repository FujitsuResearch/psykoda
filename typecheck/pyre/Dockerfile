ARG PYTHON_VERSION=3.8
FROM python:${PYTHON_VERSION}

WORKDIR /psykoda
RUN pip install --upgrade pip
RUN pip install poetry pyre-check pyright pytype
COPY pyproject.toml .
RUN poetry export -f requirements.txt -o /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN rm pyproject.toml /tmp/requirements.txt
