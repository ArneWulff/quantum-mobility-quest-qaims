# **Quantum-Assisted Stacking Sequence Retrieval and Laminated Composite Design**

This repository contains code accompanying our submission to the **Airbus/BMW Quantum Computing Challenge** ([Quantum Mobility Quest](https://qcc.thequantuminsider.com/)). We are participating in the **Golden App** category with our entry titled: **"Quantum-Assisted Stacking Sequence Retrieval and Laminated Composite Design."**

In our submission, we explore a quantum approach to solving the stacking sequence retrieval (SSR) problem for fiber-reinforced composite materials. SSR is a critical part of a prevalent bi-level optimization strategy for optimizing the mechanical characteristics of laminated composites - materials that are extensively used in the transportation sector due to their high specific strength and could play a critical role in achieving climate-neutral mobility. Our approach applies **Density-Matrix Renormalization Group (DMRG)** and **Filtering Variational Quantum Eigensolver (F-VQE)** algorithms to tackle SSR, meeting specific stiffness and strength requirements while adhering to manufacturing constraints.

## About QAIMS

Our team, **QAIMS** ([QAIMS at TU Delft](https://www.tudelft.nl/lr/qaims)), is based at TU Delft’s Faculty of Aerospace Engineering. We are an interdisciplinary group of researchers investigating the potential of innovative computational methods such as quantum computing and AI to solve
difficult problems in aerospace materials and structures.

## Contents of This Repository

The submission report is included in the root directory of this repository under the filename [`QuantumMobilityQuestSubmissionQAIMS.pdf`](QuantumMobilityQuestSubmissionQAIMS.pdf).

This repository includes two main subdirectories for the quantum algorithms explored in this project:

- [**`fvqe_python`**](fvqe_python/): Contains Python code for F-VQE in the `ssr_with_fvqe` folder, and a demonstration notebook [`demo_ssr_with_fvqe.ipynb`](fvqe_python/demo_ssr_with_fvqe.ipynb) which guides users through the essential functions for setting up and running F-VQE for SSR.

- [**`dmrg_julia`**](dmrg_julia/): Contains Julia code for DMRG in the `ssr_with_dmrg` folder, with a demonstration file [`demo_ssr_with_dmrg.jl`](dmrg_julia/demo_ssr_with_dmrg.jl) that demonstrates the setup and execution of DMRG for SSR. The demo file includes steps to implement a bias on nearest-neighbor interactions, which controls the dispersion and clustering of same-angle plies, and can be done using the existing code base. The folder `dmrg_julia` should include all necessary Julia package files to allow users to run the DMRG algorithm for SSR using `using SSRWithDMRG`.

Furthermore, we included the HDF5 files with the set of target lamination parameters for stacking sequences with conventional ply angles that we employed throughout most of our experiments. These can be found in the folder [**`target_lamination_parameters`**](target_lamination_parameters/).

## Problem Overview

The optimization of laminate composites for lightweight aerospace and automotive structures requires a robust method for configuring composite layers, also known as stacking sequences, to meet target mechanical properties. Lamination parameters, which aggregate the effects of multiple layers, serve as the primary optimization variables in a bi-level optimization framework. Given optimal lamination parameters from the first level of optimization, stacking sequence retrieval (SSR) searches for the corresponding stacking sequence in the second step. This task is combinatorial and computationally challenging due to discrete ply-angle choices and various manufacturing constraints. In our submission, we explore the potential of quantum algorithms for this problem. See also our previous publication:  
[A. Wulff *et al.* (2024), doi: 10.1016/j.cma.2024.117380](https://doi.org/10.1016/j.cma.2024.117380)

### Objectives and Algorithms

In this work, we focus on two main objectives:
1. **Finding a stacking sequence that minimizes the distance to target lamination parameters**
2. **Finding a stacking sequence with fixed ply-angle counts that maximizes the buckling factor**

We use statevector simulations of the quantum algorithm **F-VQE** to demonstrate our approach for a moderate laminate ply count. Additionally, we show its scalability to a large ply count using the classical tensor network algorithm **DMRG**.

### Constraints Considered

Our implementation includes essential manufacturing constraints:
- **Disorientation constraint** (limits angle difference between adjacent plies),
- **Contiguity constraint** (limits consecutive identical angles),
- **Balanced laminate condition** (equal counts for specific ply angles), and
- **10% rule** (ensures each ply angle has a minimum presence in the stack).

## Running the Code

1. **Requirements**:  
   The Python code, tested with Python version 3.11, has the following dependencies:
   - NumPy (tested with version 2.1.2)
   - h5py (tested with version 3.11.0)  

   The Julia code, tested with Julia version 1.8.5, requires the following packages:
   - ITensors.jl (tested with version 0.6.19)
   - ITensorsMPS.jl (tested with version 0.2.5)
   - HDF5.jl (tested with version 0.17.2)
   - KrylovKit.jl (tested with version 0.8.1)

2. **Getting started**:
   - For details on the implemented algorithms, read the submission report [`QuantumMobilityQuestSubmissionQAIMS.pdf`](QuantumMobilityQuestSubmissionQAIMS.pdf).
   - For **Python/F-VQE**, navigate to [`fvqe_python`](fvqe_python/), install the dependencies, and review [`demo_ssr_with_fvqe.ipynb`](fvqe_python/demo_ssr_with_fvqe.ipynb) to get an overview of the most important classes and methods.
   - For **Julia/DMRG**, navigate to [`dmrg_julia`](dmrg_julia/), activate the environment in [`ssr_with_dmrg`](dmrg_julia/ssr_with_dmrg/), and review [`demo_ssr_with_dmrg.jl`](dmrg_julia/demo_ssr_with_dmrg.jl) to get an overview of most important structs and functions. 

## License and Disclaimer

This code is licensed under the **Apache 2.0 License**. As it represents experimental work, it is provided "as-is" without warranties or responsibility for any potential issues arising from its use.

## Acknowledgments

We extend our thanks to the **Airbus/BMW Quantum Computing Challenge** organizers for the opportunity to showcase our work in the **Golden App** category, and to our collaborators at TU Delft.

[![TUD and QAIMS logos](img/logos_tud_qaims.png#gh-light-mode-only)](https://www.tudelft.nl/lr/qaims)
[![TUD and QAIMS logos](img/logos_white_tud_qaims.png#gh-dark-mode-only)](https://www.tudelft.nl/lr/qaims)