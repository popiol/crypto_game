#!/bin/bash

commands="
cd /home/popiol/deployments/crypto_game
git fetch
git merge origin/wip
git push
"

/bin/bash -c "$commands"
ssh crypto2 /bin/bash -c "$commands"
ssh crypto3 /bin/bash -c "$commands"
