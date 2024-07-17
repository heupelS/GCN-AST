# GCN-AST: A Movie Recommendation System based on Graph Convolutional Networks and Augmented Self-Training

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This is the implementation of a master thesis project, which aims to build a movie recommendation system based on graph convolutional networks (GCN) and augmented self-training (AST). The project is implemented in Python using PyTorch. It is built on top of [RecBole](https://github.com/RUCAIBox/RecBole), which is a comprehensive and efficient library for recommendation algorithms. The following components are part of the framework:
- **SBGA**: A self-training algorithm that uses a graph-based approach to generate pseudo-labels for unlabeled data. Implemented in [`augmented_dataset.py`](https://github.com/heupelS/ANS-Recbole/blob/main/recbole/data/dataset/augmented_dataset.py).
- **LightGCN**: A lightweight graph convolutional network model for collaborative filtering. Implemented in [`lightgcn.py`](https://github.com/yourusername/ANS-Recbole/blob/main/recbole/model/general_recommender/lightgcn.py).
- **ANS**: Augmented node sampling to improve the training process. Implemented in [`augmented_node_sampler.py`](https://github.com/yourusername/ANS-Recbole/blob/main/recbole/sampler/augmented_node_sampler.py).
- **PSST**: Pseudo Supervised Self-Training to leverage pseudo-labels effectively. Implemented in [`psst_trainer.py`](https://github.com/yourusername/ANS-Recbole/blob/main/recbole/trainer/psst_trainer.py).


## Installation

