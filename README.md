# SCUNO (Spherical Continuous-time U-shaped Neural Operator)
This is the code repository of the paper ["Conditioning Surface Shape Processes with Neural Operators"](https://link.springer.com/chapter/10.1007/978-3-032-03918-7_20).


## Overview
This project use the score-matching and time-reversal to simulate the conditioned shape surface processes.

### Shape surface modeling

We model shape as immersions $\mathrm{Imm}(\mathbb S^2, D)/\mathrm{Diff}(D)$.

Landmarks as set of evaluations: $\{s(\xi_i);\xi_i\in\mathbb{S}^2,i = 1,\cdots, N\}$.

$X_t : \xi \mapsto \phi_t(s_0)(\xi) - s_0(\xi)$.



### Shape processes:

$\mathrm d X_t = Q^{1/2}X_t \circ \mathrm d W_t$, 

where $Q^{1/2}(f)(\xi)=\int_{D}k(\phi_t(s)(\xi),\zeta)f(\zeta)\mathrm d \zeta = \int_D k (X_t(\xi)+s(\xi),\zeta)f(\zeta)\mathrm d \zeta$.

The $\xi$ are the evaluation points on shape surface $s$ and the shape differences $X_t=s_t-s_0$. $\zeta$ are the background noise points. $D$ is the noise domain.

In practice, the kernel $k$ we choose the gaussian kernel:

$k(x,y) = \kappa_{\alpha} \exp (-\frac{\lVert x-y \rVert^2}{2\kappa_\sigma^2})$, where the $\kappa_\alpha$ is the noise amplitude and the $\kappa_\sigma$ is the spatial correlation scale.
### Denoising Score matching 
We simulate conditioned shape processes through simulation of time-reversed diffusion processes via denoising score matching, where SCUNO is employed to learn the score operator.

### Spherical harmonic decomposition on the shape surfaces

Model the shape as $s(\xi) = (s^x(\xi),s^y(\xi),s^z(\xi))$, and apply spherical harmonic decompostion on each channel.

We use the package `S2FFT`: [S2FFT: differentiable and accelerated spherical transforms](https://astro-informatics.github.io/s2fft/) to apply differenciable and CUDA acceleratable spherical harmonic transforms. ([Price & McEwen 2024](https://arxiv.org/abs/2311.14670))

This project selects sampling methods based on those supported by `S2FFT`. Currently, the [Gauss-Legendre](https://link.springer.com/article/10.1007/BF02519350) sampling method is employed, with plans to support additional `S2FFT`-compatible sampling methods in the future.


## Installation

### Prerequisites
- Anaconda or Miniconda
- CUDA 12.X (for GPU support)
- Git

### Clone the repository
```bash
git clone https://github.com/MateosCz/SCUNO.git
cd SCUNO
```
### Environment Setup

We provide a conda environment file for easy setup:
```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate scuno
```
## Project Structure
```bash
.
├── checkpoints/ 
├── data/
│   ├── morph/
│   └── test_meshes/
├── src/
│   ├── dataGenerator/ #data generators
│   ├── experiments/ #experiments, jupyter notebooks
│   ├── models/ #SCUNO structures
│   ├── stochastics/ #SDE related
│   ├── training/ #trainer and losses
│   └── utils/ #helper funcitons and plotting function
├── .gitignore
├── environment.yml
├── LICENSE
└── README.md
```

## Key Dependencies

- **JAX** (0.4.30): For automatic differentiation and GPU acceleration
- **Flax** (0.8.5): Neural network library
- **s2fft** (1.3.0): Spherical Fourier transforms
- **NumPy**, **SciPy**, **Matplotlib**: Scientific computing and visualization

## Morphology data
We use the open-source [Leporid Brain Evolution](https://www.morphosource.org/projects/000417960?locale=en) dataset to test the applicability to real-world data, especially the capability to simulate evolutionary processes between two different morphologies. The model now works on real-world datasets, and we'll be releasing the experimental Jupyter notebooks soon.



## Citation

If you use this code in your research, please cite:
```bibtex
@InProceedings{10.1007/978-3-032-03918-7_20,
author="Zhou, Jingchao
and Yang, Gefan
and Sommer, Stefan",
editor="Nielsen, Frank
and Barbaresco, Fr{\'e}d{\'e}ric",
title="Conditioning Surface Shape Processes with Neural Operators",
booktitle="Geometric Science of Information",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="188--197",
abstract="We present a novel method for simulating infinite-dimensional conditional stochastic processes governing surface shape evolution. Given boundary conditions represented as spherical functions, we consider a function-valued diffusion process X with initial state {\$}{\$}X{\_}0{\$}{\$}X0, conditioned on {\$}{\$}X{\_}T{\$}{\$}XT. To address the simulation challenge, we develop a neural operator architecture leveraging spherical harmonic transforms to approximate the intractable drift term arising from Doob's h-transform. The proposed operator demonstrates discretization equivariance, enabling direct application to spherical meshes at arbitrary resolutions without architectural modifications or retraining. We validate our method on several synthetic shape evoluation scenarios.",
isbn="978-3-032-03918-7"
}


```