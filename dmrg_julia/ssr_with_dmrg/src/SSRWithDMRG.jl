"""
`SSRWithDMRG` - Stacking Sequence Retrieval (SSR) with the Density Matrix Renormalization Group (DMRG) algorithm

This module implements a DMRG-based approach to Stacking Sequence Retrieval (SSR) as developed for the 
Airbus/BMW Quantum Computing Challenge. The purpose of the module is to explore how tensor-based methods 
can be applied to retrieve optimal stacking sequences for composite laminate structures.

### Core Components
- `dmrg_experiment`: Configures and executes the DMRG optimization for SSR, including stacking sequence retrieval 
  based on specified target lamination parameters and constraints.
- Constraint Management: The module incorporates a flexible constraint framework, enabling a disorientation 
  and contiguity constraints, balanced conditions and the 10% rule.
- Functions to build MPOs corresponding to the loss function and penalty terms for the constraints.
- Structs and methods to specify a laminate and calculate lamination parameters.

### Experimental Code and Disclaimer
This code is part of an experimental research project developed for the Airbus/BMW Quantum Computing Challenge. 
The code is optimized for research and exploration and is not recommended for production use without further 
testing and validation. For additional tools, and full project context, 
please see the main repository at https://github.com/ArneWulff/quantum-mobility-quest-qaims

### Dependencies
- Julia (tested with version 1.8.5)
- ITensors.jl (tested with version 0.6.19)
- ITensorsMPS.jl (tested with version 0.2.5)
- HDF5.jl (tested with version 0.17.2)
- KrylovKit.jl (tested with version 0.8.1)

### License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. 
You may obtain a copy of the License in the LICENSE file or at http://www.apache.org/licenses/LICENSE-2.0

### Authors and Contributors
Developed by the QAIMS research group at TU Delft (https://www.tudelft.nl/lr/qaims) with Arne Wulff as corresponding author.

"""
module SSRWithDMRG

using LinearAlgebra
using ITensors
using HDF5
using KrylovKit: eigsolve
using Printf

include("laminate.jl")
include("constraints.jl")
include("mpo_loss.jl")
include("mpo_constr_nn.jl")
include("mpo_constr_knearest.jl")
include("mpo_constr_balanced.jl")
include("mpo_constr_minimum.jl")
include("dmrg_structs.jl")
include("dmrg.jl")
include("dmrg_experiment.jl")


export
    Laminate,
    num_plies,
    num_angles,
    num_funcs,
    num_weights,
    parameters,
    generate_funcs,
    generate_weights,
    ConstraintSettings,
    count_nn_constraint_violations,
    count_knearest_constraint_violations,
    count_balanced_constraint_violations,
    count_minimum_constraint_violations,
    count_constraint_violations,
    generate_loss_local_eigenvalues,
    build_mpo_loss,
    build_mpo_constr_nn,
    angles_diff,
    generate_disorientation_constraint_pq_list,
    build_mpo_constr_knearest,
    generate_plists_contiguity,
    build_mpo_constr_balanced,
    build_mpo_constr_minimum,
    build_mpo,
    DMRGResult,
    gen_bond_dims,
    gen_cutoffs,
    energy_Hsum_psi,
    result_psi_to_stack,
    dmrg_experiment
end