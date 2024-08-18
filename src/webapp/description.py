import streamlit as st

st.set_page_config(page_title="Crypto Game", layout="wide")

st.title("Crypto Game")

st.write(
    """
Cryptocurrency investment bot powered by reinforcement learning and evolutionary algorithms.

The idea of the system is to maintain a population of NN models, 
create new ones based on the previous generation, train with an RL technique,
mutate the models, evaluate and select the most promising ones.

First generation of the models were created with a simple NN model structure with one hidden layer.
Subsequent models are created by merging two models by concatenating their last hidden layers.
Then the models are mutated by randomly adding, removing, extending or shrinking the layers.

Each model is trained with one of two training strategies: learn on mistakes or learn on success.
In both cases the training set is built while simulating an investment process on the historical data.
Learning on success means that whenever a model makes a good decision, this decision is added to the training set.
Learning on mistakes means that whenever a model makes a bad decision, the opposite decision is added to the training set.
Additionally, the models are given a random noise to their weights in the training process, 
so they don't repeat the same decisions in every training epoch.
"""
)
