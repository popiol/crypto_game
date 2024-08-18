#!/bin/bash

for pid in $(pidof -x train.sh); do
    if [ $pid != $$ ]; then
        echo "Process already running"
        exit 1
    fi
done

. ../.profile

git reset --hard
git pull
mkdir -p logs/archive
timestamp=$(date '+%Y%m%d%H%M%S')
mv logs/stderr.log logs/archive/stderr_$timestamp.log 2> /dev/null
mv logs/stdout.log logs/archive/stdout_$timestamp.log 2> /dev/null
python -m src.rl_runner > logs/stdout.log 2> logs/stderr.log
