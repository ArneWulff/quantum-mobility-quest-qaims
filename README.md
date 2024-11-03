# Quantum Mobility Quest - QAIMS: Quantum-Assisted Stacking Sequence Retrieval

This repository contains code accompanying our submission to the **Airbus/BMW Quantum Computing Challenge** ([Quantum Mobility Quest](https://qcc.thequantuminsider.com/)). We are participating in the **Golden App** category with our entry titled: **"Quantum-Assisted Stacking Sequence Retrieval and Laminate Composite Design."**

In our submission, we explore a quantum approach to solving the stacking sequence retrieval (SSR) problem for fiber-reinforced composite materials. SSR is a critical challenge in optimizing laminated composite structures, particularly in the mobility sector, where advanced design solutions are essential for achieving climate-neutral mobility. Our approach applies **Density-Matrix Renormalization Group (DMRG)** and **Filtering Variational Quantum Eigensolver (F-VQE)** algorithms to tackle SSR, meeting specific stiffness and strength requirements while adhering to manufacturing constraints.

## About QAIMS

Our team, **QAIMS** ([QAIMS at TU Delft](https://www.tudelft.nl/lr/qaims)), is based at TU Delft’s Faculty of Aerospace Engineering. We are an interdisciplinary group of researchers investigating the potential of innovative computational methods such as quantum computing and AI to solve
difficult problems in aerospace material and structures.

## Contents of This Repository

This repository includes two main subdirectories for the quantum algorithms explored in this project:

- **`fvqe_python`**: Contains Python code for F-VQE in the `ssr_with_fvqe` folder, and a demonstration notebook `demo_fvqe.ipynb` which guides users through the essential functions for setting up and running F-VQE for SSR.

- **`dmrg_julia`**: Contains Julia code for DMRG in the `ssr_with_dmrg` folder, with a demonstration file `demo_ssr_with_dmrg.jl` that demonstrates the setup and execution of DMRG for SSR. This folder should include all necessary Julia package files to allow users to reproduce the results using `using SSRWithDMRG`.

## Problem Overview

The optimization of laminate composites for lightweight aerospace and automotive structures requires a robust method for configuring composite layers, also known as stacking sequences, to meet target mechanical properties. Lamination parameters, which aggregate the effects of multiple layers, serve as the primary optimization variables in a bi-level optimization framework. Given optimal lamination parameters from the first level of optimization, SSR retrieves the corresponding stacking sequence in the second step. This task is combinatorial and computationally challenging due to discrete ply-angle choices and various manufacturing constraints.

### Objectives and Algorithms

In this work, we focus on two main objectives:
1. **Minimizing distance to target lamination parameters** using Euclidean distance (both F-VQE and DMRG).
2. **Maximizing the buckling factor**, a recent addition motivated by industry feedback (here only F-VQE).

The algorithms implemented here—**DMRG** and **F-VQE**—were chosen for their ability to efficiently explore and refine configurations in large combinatorial spaces. DMRG serves as a classical tensor-based algorithm, and F-VQE is a quantum algorithm designed to filter out high-energy states, thereby improving optimization.

### Constraints Considered

Our implementation includes essential manufacturing constraints:
- **Disorientation constraint** (limits angle difference between adjacent plies),
- **Contiguity constraint** (limits consecutive identical angles),
- **Balanced laminate condition** (equal counts for specific ply angles), and
- **10% rule** (ensures each ply angle has a minimum presence in the stack).

## Running the Code

1. **Requirements**: Ensure you have the dependencies as specified in `Project.toml` (Julia) and `requirements.txt` (Python).

2. **Setup**:
   - For **Python/F-VQE**, navigate to `fvqe_python`, install dependencies, and run `demo_fvqe.ipynb`.
   - For **Julia/DMRG**, navigate to `dmrg_julia`, activate the environment in `ssr_with_dmrg`, and run `demo_ssr_with_dmrg.jl` to get started. This demo file also demonstrates how a bias on nearest-neighbor interactions to
   control the dispersion and clustering of same-angle plies can be implemented
   using the existing code base.

## License and Disclaimer

This code is licensed under the **Apache 2.0 License**. As it represents experimental work, it is provided "as-is" without warranties or responsibility for any potential issues arising from its use.

## Acknowledgments

We extend our thanks to the **Airbus and BMW Quantum Computing Challenge** organizers for the opportunity to showcase our work in the **Golden App** category, and to our collaborators at TU Delft.