# Vandermonde Neural Operator (VNO) Implementation

This repository contains a step-by-step implementation of a **Vandermonde Neural Operator (VNO)** layer. This architecture is designed to perform Fourier-based neural operations on scattered data points (irregular grids) by explicitly constructing Vandermonde matrices to map between physical space and Fourier space.

The included notebook demonstrates the inner workings of a single VNO layer, breaking down the mathematical operations into a "Lower Branch" (Global Integral Kernel) and an "Upper Branch" (Local Linear Transformation/Residual).

## 📂 File Overview

The provided notebook (`VNO_implementation.ipynb`) is divided into two main sections:

1.  **Numpy Implementation (Educational):** A loop-based, explicit construction of the Vandermonde matrix ($V$), the Adjoint matrix ($V^*$), and the Forward/Backward transforms. This section is optimized for readability to understand the exact mathematical operations occurring at each index.
2.  **PyTorch Implementation (Vectorized):** A re-implementation using standard PyTorch tensors and broadcasting. This demonstrates how to efficiently calculate the operator output using GPU-capable libraries.

## 🧮 Mathematical Logic

The code implements the update rule for a neural operator layer $v_{t} \mapsto v_{t+1}$. This architecture approximates the operator solution using the framework established for Fourier Neural Operators:

$$v_{t+1}(x) = \sigma \left( W v_t(x) + (K(\phi)v_t)(x) \right)$$

Where:

  * **$v_t$**: Input function values at scattered points $P$.
  * **$\sigma$**: Non-linear activation function (ReLU).
  * **$W$**: Linearity (Upper Branch / Residual connection).
  * **$K(\phi)$**: The Global Kernel integral operator (Lower Branch).

### The Kernel Operation ($K$)

In the VNO architecture, the kernel operation is defined via Vandermonde transforms:

$$K(\phi)v_t = V^* \cdot R_\phi \cdot V \cdot v_t$$

1.  **$V$ (Vandermonde Matrix):** Maps scattered points to Fourier modes.
      * $V_{j,k} = \sqrt{\frac{2}{n}} e^{-i \langle j, P_k \rangle}$
2.  **$R_\phi$ (Filter):** A learnable diagonal matrix (or block matrix) that mixes/filters Fourier modes. In the notebook, this is simulated as a low-pass filter.
3.  **$V^*$ (Adjoint Matrix):** Maps Fourier modes back to scattered points (Inverse transform).

## 🚀 Getting Started

### Prerequisites

  * Python 3.x
  * NumPy
  * PyTorch

### Usage

Open the Jupyter Notebook to visualize the transformation steps:

```bash
jupyter notebook VNO_implementation.ipynb
```

The code will output the intermediate states of the data (Original $\to$ Fourier Space $\to$ Filtered $\to$ Physical Space) and compare the manual Numpy calculation with the PyTorch version.

## 📚 References & Citations

If you use this logic or concepts related to the universal approximation capabilities and error bounds of Fourier Neural Operators, please cite the following foundational paper by **Siddhartha Mishra** and colleagues:

### BibTeX

```bibtex
@article{kovachki2021universal,
  author  = {Kovachki, Nikola and Lanthaler, Samuel and Mishra, Siddhartha},
  title   = {On universal approximation and error bounds for Fourier neural operators},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  url     = {http://jmlr.org/papers/v22/21-0877.html}
}
```

### APA Format

> Kovachki, N., Lanthaler, S., & **Mishra, S.** (2021). On universal approximation and error bounds for Fourier neural operators. *Journal of Machine Learning Research*

-----
