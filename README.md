# SLDENet: Adaptive Shared-Lag Decomposition for Multivariate Time Series Forecasting

This repository provides the official PyTorch implementation of our paper **"SLDENet: Adaptive Shared-Lag Decomposition for Multivariate Time Series Forecasting."**

## Introduction

SLDENet is a structure-aware neural network for multivariate time series forecasting. It aims to adaptively extract dominant temporal structures from historical sequences and exploit complementary information from different dynamic components to improve forecasting performance.

The model consists of two core modules:

- **Shared-Lag Structural Decomposition (SSD):** SSD constructs a shared-lag Gram matrix from multivariate trajectory representations and employs singular value decomposition (SVD) to extract an energy-dominant orthogonal subspace. The dominant rank is adaptively determined using the maximum spectral-gap criterion, decomposing the input sequence into dominant and residual components.

- **Dominant-Residual Enhancement (DRE):** DRE applies multi-head attention (MHA) to enhance key temporal structures and cross-variable dependencies in the dominant component, while a lightweight linear mapping extracts complementary information from the residual component.

SLDENet further integrates Reversible Instance Normalization (RevIN) and a lightweight predictor for end-to-end multivariate time series forecasting.

## Datasets

We evaluate SLDENet on **16 real-world datasets**, covering both long-term and short-term forecasting tasks.

- **Long-term forecasting:** ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Solar-Energy, and Exchange Rate.
- **Short-term forecasting:** PEMS03, PEMS04, PEMS07, PEMS08, ILI, COVID-19, NASDAQ, and SP500.

Mean Squared Error (MSE) and Mean Absolute Error (MAE) are used as evaluation metrics.

## Environment Setup

SLDENet is implemented using Python and PyTorch.

Create a Conda environment:

```bash
conda create -n SLDENet python=3.10
conda activate SLDENet
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Please install a compatible version of PyTorch and CUDA according to your computing environment.
