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
RUN python -m nltk.downloader stopwords wordnet

### Copied from Docker's Openjdk Dockerfile #####
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
# utilities for keeping Debian and OpenJDK CA certificates in sync
		ca-certificates p11-kit \
	; \
	rm -rf /var/lib/apt/lists/*

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

ENV JAVA_HOME /usr/java/openjdk-16
ENV PATH $JAVA_HOME/bin:$PATH

# backwards compatibility shim
RUN { echo '#/bin/sh'; echo 'echo "$JAVA_HOME"'; } > /usr/local/bin/docker-java-home && chmod +x /usr/local/bin/docker-java-home && [ "$JAVA_HOME" = "$(docker-java-home)" ]

# https://jdk.java.net/
# > Java Development Kit builds, from Oracle
ENV JAVA_VERSION 16-ea+5

RUN set -eux; \
	\
	savedAptMark="$(apt-mark showmanual)"; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		wget \
	; \
	rm -rf /var/lib/apt/lists/*; \
	\
	arch="$(dpkg --print-architecture)"; \
# this "case" statement is generated via "update.sh"
	case "$arch" in \
# arm64v8
		arm64 | aarch64) \
			downloadUrl=https://download.java.net/java/early_access/jdk16/5/GPL/openjdk-16-ea+5_linux-aarch64_bin.tar.gz; \
			downloadSha256=0a29b196a14e038769ff1b506e473dbae610a51c15fe965db7d192413c4c6992; \
			;; \
# amd64
		amd64 | i386:x86-64) \
			downloadUrl=https://download.java.net/java/early_access/jdk16/5/GPL/openjdk-16-ea+5_linux-x64_bin.tar.gz; \
			downloadSha256=78cd9a7c222ca3c23b74734a74011729d0260e60e65186c73806341109402e12; \
			;; \
# fallback
		*) echo >&2 "error: unsupported architecture: '$arch'"; exit 1 ;; \
	esac; \
	\
	wget -O openjdk.tgz "$downloadUrl"; \
	echo "$downloadSha256 */openjdk.tgz" | sha256sum -c -; \
	\
	mkdir -p "$JAVA_HOME"; \
	tar --extract \
		--file openjdk.tgz \
		--directory "$JAVA_HOME" \
		--strip-components 1 \
		--no-same-owner \
	; \
	rm openjdk.tgz; \
	\
	apt-mark auto '.*' > /dev/null; \
	[ -z "$savedAptMark" ] || apt-mark manual $savedAptMark > /dev/null; \
	apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
	\
# update "cacerts" bundle to use Debian's CA certificates (and make sure it stays up-to-date with changes to Debian's store)
# see https://github.com/docker-library/openjdk/issues/327
#     http://rabexc.org/posts/certificates-not-working-java#comment-4099504075
#     https://salsa.debian.org/java-team/ca-certificates-java/blob/3e51a84e9104823319abeb31f880580e46f45a98/debian/jks-keystore.hook.in
#     https://git.alpinelinux.org/aports/tree/community/java-cacerts/APKBUILD?id=761af65f38b4570093461e6546dcf6b179d2b624#n29
	{ \
		echo '#!/usr/bin/env bash'; \
		echo 'set -Eeuo pipefail'; \
		echo 'if ! [ -d "$JAVA_HOME" ]; then echo >&2 "error: missing JAVA_HOME environment variable"; exit 1; fi'; \
# 8-jdk uses "$JAVA_HOME/jre/lib/security/cacerts" and 8-jre and 11+ uses "$JAVA_HOME/lib/security/cacerts" directly (no "jre" directory)
		echo 'cacertsFile=; for f in "$JAVA_HOME/lib/security/cacerts" "$JAVA_HOME/jre/lib/security/cacerts"; do if [ -e "$f" ]; then cacertsFile="$f"; break; fi; done'; \
		echo 'if [ -z "$cacertsFile" ] || ! [ -f "$cacertsFile" ]; then echo >&2 "error: failed to find cacerts file in $JAVA_HOME"; exit 1; fi'; \
		echo 'trust extract --overwrite --format=java-cacerts --filter=ca-anchors --purpose=server-auth "$cacertsFile"'; \
	} > /etc/ca-certificates/update.d/docker-openjdk; \
	chmod +x /etc/ca-certificates/update.d/docker-openjdk; \
	/etc/ca-certificates/update.d/docker-openjdk; \
	\
# https://github.com/docker-library/openjdk/issues/331#issuecomment-498834472
	find "$JAVA_HOME/lib" -name '*.so' -exec dirname '{}' ';' | sort -u > /etc/ld.so.conf.d/docker-openjdk.conf; \
	ldconfig; \
	\
# https://github.com/docker-library/openjdk/issues/212#issuecomment-420979840
# https://openjdk.java.net/jeps/341
	java -Xshare:dump; \
	\
# basic smoke test
	javac --version; \
	java --version
#### OpenJDK Done ####


RUN set -eux; \
	apt-get update; \
	apt-get install -y gosu; \
	rm -rf /var/lib/apt/lists/*; \
	# verify that the binary works
	gosu nobody true

# create directory for the app user
ENV HOME=/home/app
RUN mkdir $HOME

RUN python -m nltk.downloader -d $HOME/nltk_data/ stopwords wordnet

ENV MALLET_BIN_DIRECTORY=$HOME/mallet-2.0.8/bin
# Prepare mallet installation
ADD http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz .
RUN tar -C $HOME -xf mallet-2.0.8.tar.gz; \
	rm mallet-2.0.8.tar.gz; \
	ls $MALLET_BIN_DIRECTORY/mallet # Check the mallet executable is there

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

# chown all the files to the app user
RUN chown -R app:app $HOME

# Needed because all the python imports look like
# from flask_app import ...
# and not from backend.flask_app import ...
WORKDIR $APP_HOME/backend/

ENTRYPOINT [ "../entrypoint.sh" ]
CMD [ "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_app.app:create_app()" ]
