# =====================================================================
# Demonstration of Julia Code for Stacking Sequence Retrieval with DMRG
# =====================================================================

using Pkg
Pkg.activate("ssr_with_dmrg")

using Dates
using ITensors: siteinds, randomMPS
using SSRWithDMRG

let
    # 🔷 laminate definition
    num_plies = 10
    angles = [0,45,90,-45]
    num_angles = length(angles)
    symmetric = true
    deg = true
    weight_types = "AD"
    func_types = ("cos2","sin2","cos4")

    # 🔷 create laminate object
    laminate = Laminate(num_plies,angles,symmetric,deg,weight_types,func_types)

    # 🔷 set target lamination parameters
    target_stack = [1, 1, 2, 3, 2, 1, 4, 3, 4, 1]
    target_parameters = parameters(laminate, target_stack)

    # 🔷 Index for numbering experiments (e.g., for different target parameters)
    # This index is added to the filename, before the ".hdf5"
    sample_idx = 1

    # 🔷 specify constraints
    constraints = ConstraintSettings(
        laminate;
        disor_penalty = 1.0/num_plies, 
        disor_dist = 46,
        disor_angles = angles,
        contiguity_penalty = 0.5/num_plies,
        contiguity_dist = 5,
        perc_penalty = 0.2/num_plies,
        perc_min = 0.1,
        balanced_penalty = 0.2/num_plies,
        balanced_angles = (2,4)
    )

    # 🔷 specify DMRG setting
    num_sweeps = 20
    max_bond_dim = 8
    sweep_sequence = "L"

    # 🔷 specify filelocation and filename
    filelocation = "path\\to\\folder\\"  # insert path to folder here
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "test_dmrg_$timestamp"

    # 🔷 Define site indices and build MPO
    sites = siteinds("Qudit", num_plies; dim = num_angles)
    mpo = build_mpo(sites, laminate, target_parameters, constraints)

    # 🔷 Initial MPS
    num_tries = 5
    psi0_list = [randomMPS(sites, 2) for _ in 1:num_tries]


    # 🔻 Run DMRG experiment
    dmrg_experiment(
        filelocation, filename, sample_idx, laminate,
        target_parameters, constraints, sites, mpo,
        psi0_list, num_sweeps, sweep_sequence, max_bond_dim
    )


    # ==========================================
    # Controling the dispersion and clustering
    # of same angle plies with a bias on nearest
    # neighbor interactions
    # ==========================================

    # 🔷 Specify constraints without the contiguity constraint
    # Include the disorientation constraint (optionally with a penalty
    # of `disor_penalty = 0.0`), as it can be modified to accomodate
    # the bias.
    constraints = ConstraintSettings(
        laminate;
        disor_penalty = 1.0/num_plies,       
        disor_dist = 46,
        disor_angles = angles,
        # perc_penalty = 0.2/num_plies,      # Optional
        # perc_min = 0.1,                    # Optional
        # balanced_penalty = 0.2/num_plies,  # Optional
        # balanced_angles = (2,4)            # Optional
    )

    # 🔷 define bias
    α = -1e-4

    # 🔷 Set according element of the nearest neighbor constraint to α
    constraints.nn_q_list[CartesianIndex.(1:num_angles,1:num_angles)] .= α

    # 🔷 build MPO and continue as before
    mpo = build_mpo(sites, laminate, target_parameters, constraints)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "test_dmrg_bias_$timestamp"
    dmrg_experiment(
        filelocation, filename, sample_idx, laminate,
        target_parameters, constraints, sites, mpo,
        psi0_list, num_sweeps, sweep_sequence, max_bond_dim
    )

end
