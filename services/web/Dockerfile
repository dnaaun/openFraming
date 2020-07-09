# Using Multistage builds to make the final image smaller.
 
#### Stage one #####
FROM python:3.8-slim-buster as builder

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./backend/requirements_no_gpu.txt . 
RUN pip wheel \
	--no-cache-dir \
	--no-deps \
	--wheel-dir \
	/usr/src/app/wheels -r requirements_no_gpu.txt


### Stage two #####
FROM python:3.8-slim-buster

# create the app user
RUN groupadd -r app -g 999 &&  useradd -r -g app  -u 999 app

## install dependencies
COPY --from=builder /usr/src/app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Install Gosu
RUN set -eux; \
	apt-get update; \
	apt-get install -y gosu; \
	rm -rf /var/lib/apt/lists/*; \
# verify that the binary works
	gosu nobody true

# create directory for the app user
ENV HOME=/home/app
RUN mkdir -p $HOME

# Prepare mallet installation
ADD mallet-2.0.8.tar.gz $HOME/
ENV MALLET_BIN_DIRECTORY=$HOME/mallet-2.0.8/bin
# Prepare project data directory, this is actually a volume 
# Handled by docker-compose.yml
ENV PROJECT_DATA_DIRECTORY=$HOME/project_data_directory
# This is similarly a volume.
ENV TRANSFORMERS_CACHE_DIRECTORY=$HOME/transformers_cache_directory
# Flask env to make sure flask doesn't serve
# static files
ENV FLASK_ENV=production

# Setup th app directory
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
# copy project
COPY . $APP_HOME

# Make sure the volumes are owned by the app user
RUN mkdir -p $PROJECT_DATA_DIRECTORY && chown app:app $PROJECT_DATA_DIRECTORY
VOLUME  $PROJECT_DATA_DIRECTORY
RUN mkdir -p $TRANSFORMERS_CACHE_DIRECTORY && chown app:app $TRANSFORMERS_CACHE_DIRECTORY
VOLUME  $TRANSFORMERS_CACHE_DIRECTORY

# chown all the files to the app user
RUN chown -R app:app $HOME
# change to the app user
USER app

# Needed because all the python imports look like
# from flask_app import ...
# and not from backend.flask_app import ...
WORKDIR $APP_HOME/backend/

CMD [ "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_app.app:create_app()" ]
