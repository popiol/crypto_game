#!/bin/bash

for pid in $(pidof -x evaluate.sh); do
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
mv logs/evaluate.log logs/archive/evaluate_$timestamp.log 2> /dev/null
python -m src.rl_runner --config config/config.yml --evaluate > logs/evaluate.log 2> logs/evaluate.log
