#!/bin/bash

. ../.profile

mkdir -p logs/archive
timestamp=$(date '+%Y%m%d%H%M%S')
mv logs/stderr.log logs/archive/stderr_$timestamp.log 2> /dev/null
mv logs/stdout.log logs/archive/stdout_$timestamp.log 2> /dev/null
python -m src.rl_runner --config config/config.yml > logs/stdout.log 2> logs/stderr.log
