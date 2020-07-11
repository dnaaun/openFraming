#!/bin/bash

set -e

chown -R app:app $PROJECT_DATA_DIRECTORY
chown -R app:app $TRANSFORMERS_CACHE_DIRECTORY

gosu app:app "$@"
