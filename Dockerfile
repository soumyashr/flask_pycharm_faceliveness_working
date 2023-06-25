FROM python:3.8

ENV VIRTUAL_ENV=/venv_docker
RUN python -m venv venv_docker
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

ADD . /app

COPY requirements.txt .


RUN pip install -U pip wheel cmake

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python


# install dependencies
RUN pip install -r requirements.txt
# COPY  . .
# expose port
EXPOSE 5000
# run the application
CMD ["python", "app.py"]