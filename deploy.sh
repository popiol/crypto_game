#!/bin/bash

commands="
cd /home/popiol/deployments/crypto_game
git fetch
git merge origin/wip
git push
cd -
"

/bin/bash -c "$commands"
ssh crypto2 /bin/bash -c "$commands"
