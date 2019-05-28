# IN-pytorch

PyTorch implementation of Interaction Networks (Battaglia et al., 2016).

*Interaction Networks for Learning about Objects, Relations and Physics*
https://arxiv.org/abs/1612.00222

"Reasoning about objects, relations, and physics is central to human intelligence, and
a key goal of artificial intelligence. Here we introduce the interaction network, a
model which can reason about how objects in complex systems interact, supporting
dynamical predictions, as well as inferences about the abstract properties of the
system."

## Data generation

`python physics.py`

Automatically generates motion data for a three-particle system with arbitrary orbit, together with a video showing a sample of the data saved as "test.mp4"

![test_data](https://github.com/ToruOwO/InteractionNetwork-pytorch/blob/master/test.gif)

## Model training

`python model.py`

Trains the Interaction Network and plot loss for each epoch