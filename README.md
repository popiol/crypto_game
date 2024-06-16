# crypto_game

This is an implementation of a reinforcement learning method for a cryptocurrency trading bot.

## Basics

The whole process starts with downloading hourly crypto exchange quotes from the last week.
Those quotes are then passed to learning agents. The agents make decisions about
buying and selling crypto assets based on their ML models. Each agent has its own model.
The agents also train their models dynamically based on the recent results. 
The training repeats with the current dataset for 24 hours, 
after which a new dataset is downloaded and new models are created.

## Evolution

The models are stored on S3 and can be used as pre-trained models by next generations.
The pre-trained models can also be merged together by concatenating their last layers before the output layers.

```
model_1 = input -> layer_1_1 -> layer_1_2 -> output
model_2 = input -> layer_2_1 -> layer_2_2 -> output
model_3 = input -> (layer_1_1, layer_2_1) -> concat(layer_1_2, layer_2_2) -> layer_3_3 -> output
```

## Model structure

The input layer has shape (n_assets, n_steps, n_features).

The output layer has shape (n_assets, n_outputs).

The output values for each asset are: score, buy_price, sell_price. 
The score value represents the confidence that the asset should be bought. 
It is also used to calculate the volume to buy. 
The buy_price is the buy price limit relative to the last transaction price.
The sell_price is the sell price limit relative to the last transaction price (assuming the asset is in the portfolio).

The model structure is as follows:

* Input: (n_steps, n_assets, n_features)
* Permute dimensions: (n_steps, n_features, n_assets)
* Dense layer: (n_steps, n_features, n_assets_hidden)
* Dense layer: (n_steps, n_features, n_assets)
* Permute dimensions: (n_assets, n_steps, n_features)
* Reshape: (n_assets, n_steps * n_features)
* Dense layer: (n_assets, n_hidden)
* Output layer: (n_assets, n_outputs)

## Learning

Each model follows one of the strategies:

* learn on mistakes
    use only losing transactions and change each output value x to 1 - x

* learn on success
    randomize the model weights on prediction and learn on the best results only
