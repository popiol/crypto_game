# crypto_game

This is an implementation of a reinforcement learning method for a cryptocurrency trading bot.

## Basics

The whole process starts with downloading the current quotes from a crypto exchange.
Those quotes are then passed to learning agents. The agents make decisions about
buying and selling crypto assets based on an ML model. Each agent has its own model.
The agents also train their models based on the current portfolio value.

## Parallel processing and evolution

Each agent runs in a separate process. 
The models they train are stored on S3 and can be potentially used as pre-trained models by other processes.
The pre-trained models can also be merged together by concatenating their output layers.

