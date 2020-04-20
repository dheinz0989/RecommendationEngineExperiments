
# RecommenderEngine
This module contains minor Machine Learning experiments with regards to [Recommendation Engines](https://medium.com/voice-tech-podcast/a-simple-way-to-explain-the-recommendation-engine-in-ai-d1a609f59d97). It examines different algorithms (at the moment of writing,i.e. 20th of April 2020, only one is implemented) used for Predicting the [Click Through Rate](https://en.wikipedia.org/wiki/Click-through_rate). It uses the DeepFM module found in the [deepctr](https://pypi.org/project/deepctr/) module. 
# Prerequisites
The source code is written in [Python 3.8](https://www.python.org/). It mainly uses the  [deepctr](https://pypi.org/project/deepctr/) and [scikit-learn packages](https://scikit-learn.org/stable/).
# Installation
You can clone this repository by running:

    git clone https://github.com/dheinz0989/Utilities_Import

# Requirements
As mentioned in the introduction, two main packages are used within this repo. In order to run the script, you need to install them prior to use it. The recommended standard approach is to pip install them.

    pip install requirements.txt

# Usage
At the 20th of April 2020, only model training is implemented. You can train a model based on movie lens with the following command.

    python train.py --data_source data/ratings.csv --feature_columns userId movieId --target_columns click --model_backend deepctr --model_name DeepFMModel

# Author
Author of this repo = Dominik Heinz
Contact Information = dheinz0989@gmail.com
