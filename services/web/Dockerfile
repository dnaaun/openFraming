# Using Multistage builds to make the final image smaller.
 
#### Stage one #####
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime as builder

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./backend/requirements_gpu.txt . 
RUN pip wheel \
	--no-cache-dir \
	--no-deps \
	--wheel-dir \
	/usr/src/app/wheels -r requirements_gpu.txt


### Stage two #####
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

# This is needed for the openJDK copy-paste to use wget.
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends wget

######################################### Copied from Docker's Openjdk Dockerfile ###############################################
##### https://github.com/docker-library/openjdk/blob/8107dd6c8a2f42c91fd37904fdfe7d4c434ca5fc/16/jdk/buster/Dockerfile ########
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		bzip2 \
		unzip \
		xz-utils \
		\
# utilities for keeping Debian and OpenJDK CA certificates in sync
		ca-certificates p11-kit \
		\
# jlink --strip-debug on 13+ needs objcopy: https://github.com/docker-library/openjdk/issues/351
# Error: java.io.IOException: Cannot run program "objcopy": error=2, No such file or directory
		binutils \
# java.lang.UnsatisfiedLinkError: /usr/local/openjdk-11/lib/libfontmanager.so: libfreetype.so.6: cannot open shared object file: No such file or directory
# java.lang.NoClassDefFoundError: Could not initialize class sun.awt.X11FontManager
# https://github.com/docker-library/openjdk/pull/235#issuecomment-424466077
		fontconfig libfreetype6 \
	; \
	rm -rf /var/lib/apt/lists/*

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

ENV JAVA_HOME /usr/local/openjdk-16
ENV PATH $JAVA_HOME/bin:$PATH

# backwards compatibility shim
RUN { echo '#/bin/sh'; echo 'echo "$JAVA_HOME"'; } > /usr/local/bin/docker-java-home && chmod +x /usr/local/bin/docker-java-home && [ "$JAVA_HOME" = "$(docker-java-home)" ]

# https://jdk.java.net/
# >
# > Java Development Kit builds, from Oracle
# >
ENV JAVA_VERSION 16-ea+26

RUN set -eux; \
	\
	arch="$(dpkg --print-architecture)"; \
# this "case" statement is generated via "update.sh"
	case "$arch" in \
# arm64v8
		arm64 | aarch64) \
			downloadUrl=https://download.java.net/java/GA/jdk16/7863447f0ab643c585b9bdebf67c69db/35/GPL/openjdk-16_linux-aarch64_bin.tar.gz; \
			downloadSha256=5afe8561d1c6f777bf0f70bf5d994187ff607b8c4c6ff1194f849a0bb933d805			;; \
# amd64
		amd64 | i386:x86-64) \
			downloadUrl=https://download.java.net/java/GA/jdk16/7863447f0ab643c585b9bdebf67c69db/35/GPL/openjdk-16_linux-x64_bin.tar.gz; \
			downloadSha256=25f51624a17545a769e97ded5d51075ab31f9a52de925f7292cff951becf8fd2 \
			;; \
# fallback
		*) echo >&2 "error: unsupported architecture: '$arch'"; exit 1 ;; \
	esac; \
	\
	wget -O openjdk.tgz "$downloadUrl" --progress=dot:giga; \
	echo "$downloadSha256 *openjdk.tgz" | sha256sum --strict --check -; \
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
	fileEncoding="$(echo 'System.out.println(System.getProperty("file.encoding"))' | jshell -s -)"; [ "$fileEncoding" = 'UTF-8' ]; rm -rf ~/.java; \
	javac --version; \
	java --version
######################## OpenJDK Done #############################
#
# create directory for the app user
ENV HOME=/home/app
RUN mkdir $HOME


ENV MALLET_BIN_DIRECTORY=$HOME/mallet-2.0.8/bin
# Prepare mallet installation
ADD http://mallet.cs.umass.edu/dist/mallet-2.0.8.tar.gz .
RUN tar -C $HOME -xf mallet-2.0.8.tar.gz; \
	rm mallet-2.0.8.tar.gz; \
	ls $MALLET_BIN_DIRECTORY/mallet # Check the mallet executable is there


RUN set -eux; \
	apt-get update; \
	apt-get install -y gosu; \
	rm -rf /var/lib/apt/lists/*; \
	# verify that the binary works
	gosu nobody true

# create the app user
RUN groupadd -r app -g 999 &&  useradd -r -g app  -u 999 app

## install dependencies
COPY --from=builder /usr/src/app/wheels /wheels
RUN pip install --no-cache /wheels/*; \
	rm -rf /wheels
RUN python -m nltk.downloader stopwords wordnet

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
ENV FLASK_APP=flask_app/app.py
ENV FLASK_ENV=development
ENV FLASK_RUN_HOST=0.0.0.0
# CMD [ "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "flask_app.app:create_app()" ]
CMD ["flask", "run"]
