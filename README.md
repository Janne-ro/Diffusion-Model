# Diffusion Model Self-Study

This repository contains a Jupyter Notebook (`Diffusion_model.ipynb`) documenting a self-study undertaken to deepen understanding of diffusion models. The primary goal was to replicate results from the paper "[**Generative Modeling by Estimating Gradients of the Data Distribution** by Song et al. (2019)](https://proceedings.neurips.co/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html?ref=https://githubhelp.com)".

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Project Overview

This notebook explores the core concepts of diffusion models by implementing a basic diffusion model for image generation. It specifically focuses on:
* Generating a precise noise sequence.
* Defining and implementing a U-Net architecture.
* Implementing the denoising loss function described in the original paper.
* Training the model on the MNIST dataset.

## Key Features

- **Noise Sequence Generation:** Implements a specific noise sequence satisfying conditions $\sigma_1=1$, $\sigma_{10}=0.01$, and a constant ratio between consecutive noise levels.
- **U-Net Architecture:** Features an encoder with three layers and ReLU activation, and a decoder with transposed convolutions. It includes Conditional Instance Normalization (CIN) and integrates noise as a feature vector in accordance with the paper's appendix.
- **Denoising Loss Function:** Utilizes a custom denoising loss function equivalent to the one described in the paper for training the U-Net.
- **Gaussian Noise Addition:** Includes a helper function to add Gaussian noise to images.
- **MNIST Dataset Integration:** Demonstrates loading and preprocessing of the MNIST dataset for training.

## Setup and Installation

To run this notebook, you will need to have Python and Jupyter installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Janne-ro/Diffusion-Model](https://github.com/Janne-ro/Diffusion-Model)
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    The necessary libraries can be installed using pip. You can find the list of dependencies in the `Dependencies` section below. An example looks as follows:
    ```bash
    pip install torch torchvision matplotlib numpy
    ```

## Usage

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Diffusion_model.ipynb
    ```

2.  **Run the cells:** Execute the cells sequentially to understand the implementation of the diffusion model, from noise generation to model training.

## Dependencies

The following Python libraries are required to run the notebook:
- `torch`
- `torchvision`
- `torch.nn`
- `torch.nn.functional`
- `torch.optim`
- `matplotlib.pyplot`
- `math`
- `numpy`
- `random`
- `matplotlib.image`


