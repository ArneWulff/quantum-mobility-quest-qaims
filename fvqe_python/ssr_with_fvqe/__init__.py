"""
`ssr_with_fvqe` - Quantum-Assisted Stacking Sequence Retrieval (SSR) Laminate Composite Design

This module provides an implementation of the Filtering Variational Quantum Eigensolver 
(F-VQE) algorithm, developed as part of the Airbus/BMW Quantum Computing Challenge. The 
objective is to explore how quantum computing techniques can enhance stacking sequence 
retrieval (SSR) for optimizing composite laminate structures. The F-VQE algorithm 
specifically addresses the unique challenges of composite stacking by integrating 
domain-specific constraints and objectives such as laminate parameter alignment and 
buckling factor maximization.

Core Components:
    - `fvqe_experiment` and `fvqe_experiment_buckling`: Configure and run F-VQE experiments 
      for either laminate parameter search or buckling factor maximization in composite plates.
    - Variational Quantum Circuits: Parameterized quantum circuits based on RY and CNOT gates, 
      along with permutation-based SSR circuits, provide flexible configurations for SSR tasks.
    - Material and Structural Calculations: Includes lamination parameter calculations, 
      ply-level constraints, and material stiffness/buckling formulas.

Dependencies:
    - Python (tested with version 3.11)
    - NumPy (tested with version 2.1.2)
    - h5py (tested with version 3.11.0)

License:
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License in the LICENSE file of the
    repository or at http://www.apache.org/licenses/LICENSE-2.0

Author:
    QAIMS research group at TU Delft (https://www.tudelft.nl/lr/qaims)
    Corresponding author: Arne Wulff

Note:
    This repository contains experimental code developed for a quantum computing competition 
    and is shared for research and evaluation purposes. For full context, additional tools, 
    and related code, please see the main repository at:
    https://github.com/ArneWulff/quantum-mobility-quest-qaims

Disclaimer:
    As experimental code, this module is not guaranteed for production use and is best suited 
    for research or further development by those familiar with the relevant concepts.
"""

__version__ = "0.1.0"

# Laminate definition
from .laminate import Laminate, create_laminate, generate_funcs, generate_weights

# Parameterized states
from .parameterized_state import ParameterizedState, parameterized_hwe_state
from .parameterized_state_perm import ParameterizedStatePerm, parameterized_perm_state

# Core F-VQE functionality
from .fvqe import run_fvqe, FvqeResults, get_filter_fn
from .fvqe_experiment import fvqe_experiment, FvqeOptions, hdf5_to_fvqe_result
from .fvqe_experiment_buckling import fvqe_experiment_buckling

# Utilities and additional configurations
from .encoding import (
    stack_to_state, state_to_stack, 
    state_to_dense, dense_to_state, 
    stack_to_dense, dense_to_stack
)
from .constraints import ConstraintSettings

# Control the accessible API when using `from ssr_with_fvqe import *`
__all__ = [
    "Laminate",
    "create_laminate",
    "generate_funcs",
    "generate_weights",
    "ParameterizedState",
    "parameterized_hwe_state",
    "ParameterizedStatePerm",
    "parameterized_perm_state",
    "run_fvqe",
    "FvqeResults",
    "get_filter_fn",
    "fvqe_experiment",
    "fvqe_experiment_buckling",
    "FvqeOptions",
    "hdf5_to_fvqe_result",
    "stack_to_state",
    "state_to_stack",
    "state_to_dense",
    "dense_to_state",
    "dense_to_stack",
    "stack_to_dense",
    "ConstraintSettings",
]