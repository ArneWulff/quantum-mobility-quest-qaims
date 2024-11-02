"""
fvqe_experiment_buckling.py

This module provides functions to set up and run an F-VQE experiment for optimizing the 
buckling factor of laminated composite plates. It includes material property functions 
and experiment configurations tailored to the buckling maximization objective.

Functions:
    - `calc_q`: Computes the stiffness matrix `Q` for an orthotropic ply, based on 
      material properties such as Young's moduli, shear modulus, and Poisson's ratio.
    
    - `buckling_formula`: Calculates the buckling factor `λ_B` for a rectangular composite 
      plate under specified bi-axial loading conditions.
    
    - `tsai_pagano_invariants`: Derives Tsai-Pagano invariants from the stiffness matrix 
      `Q`, essential for calculating laminate stiffness matrices.
    
    - `gen_gammas`: Generates Γ matrices used in bending stiffness calculations.
    
    - `d_matrix`: Computes the D matrix for laminate bending stiffness, incorporating ply
      lamination parameters and material invariants.
    
    - `fvqe_experiment_buckling`: Configures and runs an F-VQE experiment to maximize the 
      buckling factor `λ_B` for a laminated composite plate. Constraints, such as ply-angle 
      counts, may be enforced either through direct parameter encoding or through penalty 
      terms, depending on the circuit type.
    
Usage:
    This module is inteded to be used to optimize the stacking sequence of laminated plates 
    for maximum buckling resistance. The F-VQE algorithm adjusts ply angles while adhering to 
    structural constraints. Results are saved in an HDF5 file for further analysis.

Notes:
    - The `fvqe_experiment_buckling` function allows for two approaches to enforce ply-angle
      counts:
        - For `vqc == 'perm'`, ply angles are enforced through initial state encoding.
        - For other circuits, a penalty term is applied to penalize deviations from 
          specified ply-angle counts.
    - The `material_opts` dictionary can be used to specify custom material properties 
      and plate dimensions for buckling calculations.
    - The resulting HDF5 file contains detailed information on material properties, 
      constraints, encoding scheme, and experiment results.
"""


import numpy as np
from random import shuffle
import itertools
from typing import Optional, Callable, Sequence
from numpy.typing import NDArray
from os.path import isfile
import h5py
from .typing import Encoding, FilterFunc, CircuitParams, Stack, Parameters
from .laminate import Laminate
from .encoding import stack_to_state, convert_func_qd_to_qb, state_to_stack, convert_func_to_dense, dense_to_stack, state_to_dense
from .constraints import ConstraintSettings
from .fvqe import FvqeResults, run_fvqe
from .fvqe_experiment import hdf5_to_fvqe_result, FvqeOptions

def calc_q(E1,E2,G12,nu12):
    """
    Calculates the stiffness matrix Q for an orthotropic ply, based on intrinsic
    material properties.

    The stiffness components are derived from the longitudinal (`E1`) and transverse
    (`E2`) Young's moduli, shear modulus (`G12`), and Poisson's ratio (`nu12`).
    This matrix forms the basis for calculating Tsai-Pagano invariants.

    Args:
        E1 (float): Longitudinal Young's modulus, in GPa.
        E2 (float): Transverse Young's modulus, in GPa.
        G12 (float): Shear modulus, in GPa.
        nu12 (float): Poisson's ratio (dimensionless).

    Returns:
        np.ndarray: Stiffness matrix Q with entries [Q11, Q12, Q22, Q66].
    """
    nu21 = nu12 * E1/E2
    denom = 1/(1 - nu12 * nu21)
    Q11 = E1 * denom
    Q12 = nu12 * Q11 * denom
    Q22 = E2 * denom
    Q66 = G12
    return np.array([Q11, Q12, Q22, Q66])
    

def buckling_formula(D11: float, D12: float, D22: float, D33: float, 
                     m: int, n: int, Nx: float, Ny: float, a: float, b: float):
    """
    Calculates the buckling factor λ_B for a rectangular composite plate
    under bi-axial loading, using a simplified buckling formula.

    Args:
        D11, D12, D22, D33 (float): Components of the laminate D matrix.
        m, n (int): Half-wave numbers in the x and y directions (dimensionless).
        Nx, Ny (float): In-plane stress resultants in kN/m.
        a, b (float): Plate dimensions, in meters.

    Returns:
        float: The buckling factor λ_B, with buckling occurring if 0 < λ_B < 1.
    """
    return np.pi**2 * (
        D11 * (m/a)**4 + 2 * (D12 + D33) * (m/a)**2 * (n/b)**2
        + D22 * (n/b)**4
    ) / ((m/a)**2 * Nx + (n/b)**2 * Ny)

def tsai_pagano_invariants(Q: NDArray):
    """
    U1 = (3 Q11 + 3Q22 + 2Q12 + 4Q66)/8
    U2 = (Q11 - Q22)/2
    U3 = (Q11 + Q22 - 2Q12 - 4Q66)/8
    U4 = (Q11 + Q22 + 6Q12 - 4Q66)/8
    U5 = (Q11 + Q22 - 2Q12 + Q66)/8
    """
    trafo = np.array([[3, 3, 2, 4],
                      [4,-4, 0, 0],
                      [1, 1,-2,-4],
                      [1, 1, 6,-4],
                      [1, 1,-2, 4]])/8
    return trafo @ Q


def gen_gammas(U: Sequence):
    """
    Generates the Γ matrices for calculating laminate stiffness matrices.

    Args:
        U (Sequence): Sequence of Tsai-Pagano invariants [U1, U2, U3, U4, U5].

    Returns:
        np.ndarray: Array of 3x3 Γ matrices.
    """
    U1,U2,U3,U4,U5 = U
    U22 = U2/2
    return np.array([
        [[ U1,  U4,  0],
         [ U4,  U1,  0],
         [  0,   0, U5]],
        [[ U2,   0,  0],
         [  0, -U2,  0],
         [  0,   0,  0]],
        [[  0,   0,U22],
         [  0,   0,U22],
         [U22, U22,  0]],
        [[ U3, -U3,  0],
         [-U3,  U3,  0],
         [  0,   0,-U3]],
        [[  0,   0, U3],
         [  0,   0,-U3],
         [ U3, -U3,  0]]
    ], dtype=float)

def d_matrix(lp_d: NDArray, h: float, U: Sequence[float]):
    """
    Calculates the D matrix for laminate bending stiffness, using the ply's
    lamination parameters and material invariants.

    Args:
        lp_d (NDArray): Lamination parameters for bending.
        h (float): Ply thickness, in meters.
        U (Sequence[float]): Tsai-Pagano invariants for the material.

    Returns:
        np.ndarray: The laminate D matrix (3x3), used in buckling calculations.
    """
    gammas = gen_gammas(U)
    return h**3 / 12 * (gammas[0] + np.sum(lp_d[:,None,None] * gammas[1:(len(lp_d)+1)], axis=0))
    


def fvqe_experiment_buckling(
    filepath: str,
    laminate: Laminate,
    ply_angle_numbers: Sequence[int],
    max_buckling_factor: float,
    encoding: Encoding,
    fvqe_options: FvqeOptions,
    number_constraint_penalty: float = 1.,
    number_constraint_log: bool = False,
    constraint_settings: Optional[ConstraintSettings] = None,
    vqc: str = 'hwe',
    material_opts: Optional[dict[str,float]] = None,
    print_info: bool = False,
    resume_filepath: Optional[str] = None
):
    """
    Runs the F-VQE algorithm for maximizing the buckling factor in laminated composites.

    This function is tailored to maximize the buckling factor `λ_B` of a laminated composite
    plate, subject to ply-angle constraints. The algorithm optimizes the stacking sequence
    for a given laminate configuration while ensuring ply-angle counts according to
    `ply_angle_numbers`. Ply-angle enforcement depends on the type of variational
    quantum circuit (`vqc`) specified:
    
    - For `vqc == 'perm'`: Enforces ply-angle counts directly through parameterized 
      partial swap gates, eliminating the need for a penalty term.
    - For other circuits: Adds a penalty function (`number_constraint`) to encourage 
      ply-angle count compliance, with an optional logarithmic scaling (`number_constraint_log`) 
      to avoid overshadowing the primary objective.

    Args:
        filepath (str): Path to save the results as an HDF5 file. Raises an error if the file exists.
        laminate (Laminate): The laminate instance defining ply functions and weight matrices.
        ply_angle_numbers (Sequence[int]): Sequence specifying the target count of each ply angle.
        max_buckling_factor (float): Maximum allowable buckling factor for optimization.
        encoding (Encoding): Encoding scheme for the ply angle sequences.
        fvqe_options (FvqeOptions): Options for running F-VQE, encapsulated in an options object.
        number_constraint_penalty (float, optional): Penalty weight for ply-angle count constraint.
            Ignored if `vqc == 'perm'`.
        number_constraint_log (bool, optional): Whether to apply a logarithmic scale to the 
            penalty for ply-angle counts. Ignored if `vqc == 'perm'`. Defaults to False.
        constraint_settings (Optional[ConstraintSettings], optional): Additional constraints, such
            as disorientation or percent rules, to be added to the energy function. Defaults to None.
        vqc (str, optional): Type of variational quantum circuit to use ('hwe', 'mera', 'merarep', or 'perm').
            Defaults to 'hwe'.
        material_opts (Optional[dict[str, float]], optional): Dictionary with material properties
            used to calculate the stiffness matrix `Q`:
                - `E1`, `E2`: Young's moduli in the longitudinal and transverse directions, in GPa.
                - `G12`: Shear modulus, in GPa.
                - `nu12`: Poisson's ratio (dimensionless).
                - `a`, `b`: Plate dimensions, in meters.
                - `Nx`, `Ny`: In-plane stress resultants in kN/m.
                - `h`: Ply thickness, in meters.
                - `m`, `n`: Half-wave numbers in the x and y directions (dimensionless).
            Defaults to typical values if not provided.
        print_info (bool, optional): Whether to print information during optimization.
            Defaults to False.
        resume_filepath (Optional[str], optional): Path to an HDF5 file to resume optimization from
            previous results. Defaults to None.

    Returns:
        FvqeResults: The results of the F-VQE optimization process, including energies, parameters,
            taus, gradients, and timestamps.

    Notes:
        - The primary objective function is the buckling factor, calculated with a
          plate-buckling formula, subject to ply-angle constraints.
        - If `vqc == 'perm'`, initial states representing possible permutations are
          generated from `ply_angle_numbers`, avoiding the need for ply-angle penalties.
        - The resulting HDF5 file contains properties, material options, and constraints.
    """
    if isfile(filepath):
        raise ValueError(f"File {filepath} already exists.")

    num_qubits = laminate.num_plies * len(encoding[0])

    default_material_opts = dict(
            E1 = 177.,  # GPa
            E2 = 10.8,  # GPa
            G12 = 7.6,  # GPa
            nu12 = 0.27,
            a = 1.,
            b = 1.,
            Nx = 2.,
            Ny = 1.,
            h = 1.,
            n = 1,
            m = 1
    )
    if material_opts is not None:
        default_material_opts.update(material_opts)
    material_opts = default_material_opts

    q = calc_q(
        material_opts['E1'],
        material_opts['E2'],
        material_opts['G12'],
        material_opts['nu12'],
    )

    u = tsai_pagano_invariants(q)
    
    def energy_buckling(stack):
        lp_d = laminate.weights[1] @ laminate.funcs[stack]
        d_mat = d_matrix(
            lp_d, material_opts['h'], u
        )
        return max_buckling_factor - buckling_formula(
            d_mat[0,0],
            d_mat[0,1],
            d_mat[1,1],
            d_mat[2,2],
            material_opts['m'],
            material_opts['n'],
            material_opts['Nx'],
            material_opts['Ny'],
            material_opts['a'],
            material_opts['b']
        )

    perm_initial_state = None
    if vqc == 'perm':
        def number_constraint(stack):
            return 0.
        potential_initial_stacks = np.array([
            [i for i, count in zip(p,ply_angle_numbers[list(p)]) for _ in range(count)]
            for p in itertools.permutations(np.arange(len(ply_angle_numbers)))
        ])
        potential_initial_energies = [
            energy_buckling(stack) for stack in potential_initial_stacks
        ]
        initial_stack_idx = np.argmin(potential_initial_energies)
        initial_stack = potential_initial_stacks[initial_stack_idx]
        perm_initial_state = stack_to_state(initial_stack, encoding)
    elif number_constraint_log:
        def number_constraint(stack):
            """ log2(1+x): 0 -> log2(1) = 0,  1 -> log2(2) = 1 """
            return number_constraint_penalty * np.log2(1 + np.sum(
                abs(np.sum(stack == s) - pan) for s,pan in enumerate(ply_angle_numbers)
            ))
    else:
        def number_constraint(stack):
            # return 0.
            return number_constraint_penalty * np.sum(
                abs(np.sum(stack == s) - pan) for s,pan in enumerate(ply_angle_numbers)
            )

    
    
    if constraint_settings is None:
        energy_fn = lambda x: energy_buckling(dense_to_stack(x, num_qubits, encoding)) + number_constraint(dense_to_stack(x, num_qubits, encoding))
    else:
        energy_fn = lambda x: (
            energy_buckling(dense_to_stack(x, num_qubits, encoding)) + number_constraint(dense_to_stack(x, num_qubits, encoding)) 
            + constraint_settings.penalty(
                dense_to_stack(x, num_qubits, encoding)
            )
        )

    if resume_filepath is not None:
        previous_results = hdf5_to_fvqe_result(resume_filepath)
    else:
        previous_results = None

    results = run_fvqe(
        num_qubits,
        fvqe_options.num_reps,
        energy_fn,
        fvqe_options.filter_fn,
        x0=fvqe_options.x0,
        shots=fvqe_options.shots,
        maxiter=fvqe_options.maxiter,
        learning_rate=fvqe_options.learning_rate,
        max_tau=fvqe_options.max_tau,
        tau_increment=fvqe_options.tau_increment,
        adapt_tau_increment=fvqe_options.adapt_tau_increment,
        par_inds_fn=fvqe_options.par_inds_fn,
        reverse_par_inds=fvqe_options.reverse_par_inds,
        g_c=fvqe_options.g_c,
        renormalize_gc=fvqe_options.renormalize_gc,
        vqc=vqc,
        previous_results=previous_results,
        print_info=print_info,
        perm_initial_state=perm_initial_state
    )

    counts_array_shape = (len(results.counts), max(len(c) for c in results.counts))
    counts_array_states = np.zeros(counts_array_shape, dtype=f'S{num_qubits}')
    counts_array_vals = np.zeros(counts_array_shape, dtype=int)
    for j, counts_dict in enumerate(results.counts):
        co_st, co_va = zip(*counts_dict.items())
        co_st = np.array(co_st, dtype=f'S{num_qubits}')
        co_va = np.array(co_va, dtype=int)
        counts_array_states[j, :len(co_st)] = co_st
        counts_array_vals[j, :len(co_va)] = co_va

    # create hdf5
    with h5py.File(filepath, "w") as file:
        props = file.create_group("Properties")

        props_lam = props.create_group("Laminate")
        props_lam.attrs["num_plies"] = laminate.num_plies
        props_lam.attrs["num_angles"] = laminate.num_angles
        props_lam.attrs["num_weights"] = laminate.num_weights
        props_lam.attrs["num_funcs"] = laminate.num_funcs

        props.create_dataset('ply_angle_numbers',data=ply_angle_numbers)
        props.attrs["number_constraint_penalty"] = number_constraint_penalty
        props_m = props.create_group('material_opts')
        for key,val in material_opts.items():
            props_m.attrs[key] = val
        props.attrs["max_buckling_factor"] = max_buckling_factor
        props.attrs["number_constraint_log"] = number_constraint_log
        if perm_initial_state is not None:
            props.attrs["perm_initial_state"] = perm_initial_state
            props.create_dataset("perm_initial_stack", data=np.array(initial_stack))

        props_c = props.create_group("Contraints")
        if constraint_settings is not None:
            for constraint, val in constraint_settings.constraints.items():
                props_c.attrs[f"{constraint}_penalty"] = val
                match constraint:
                    case "disorientation":
                        props_c.create_dataset(
                            "disorientation_matrix", data=np.array(constraint_settings.disorientation_matrix)
                        )
                    case "contiguity":
                        props_c.attrs["contiguity_distance"] = constraint_settings.contiguity_distance
                    case "balanced":
                        props_c.create_dataset(
                            "balanced_angles", data=np.array(constraint_settings.balanced_angles)
                        )
                    case "percent":
                        props_c.create_dataset(
                            "percent_rule", data=np.array(constraint_settings.percent_rule)
                        )
                        props_c.create_dataset(
                            "percent_rule_min_plies", data=np.array(constraint_settings.percent_rule_min_plies)
                        )

        props_q = props.create_group("Encoding")
        props_q.attrs["num_qubits"] = num_qubits
        props_q.create_dataset("encoding", data=np.array(encoding))

        props_fvqe = props.create_group("FVQE")
        props_fvqe.attrs['num_reps'] = fvqe_options.num_reps
        props_fvqe.attrs['filter_fn'] = fvqe_options.filter_fn_name
        props_fvqe.attrs['shots'] = fvqe_options.shots
        props_fvqe.attrs['maxiter'] = fvqe_options.maxiter
        props_fvqe.attrs['learning_rate'] = fvqe_options.learning_rate
        props_fvqe.attrs['max_tau'] = fvqe_options.max_tau
        props_fvqe.attrs['tau_increment'] = fvqe_options.tau_increment
        props_fvqe.attrs['adapt_tau_increment'] = fvqe_options.adapt_tau_increment
        props_fvqe.attrs['reverse_par_inds'] = fvqe_options.reverse_par_inds
        props_fvqe.attrs['g_c'] = fvqe_options.g_c
        props_fvqe.attrs['renormalize_gc'] = fvqe_options.renormalize_gc

        res = file.create_group("Results")
        res.create_dataset("energies", data=np.array(results.energies))
        res.create_dataset("parameters", data=np.array(results.parameters))
        res.create_dataset("taus", data=np.array(results.taus))
        res.create_dataset("gradients", data=np.array(results.gradients))
        res.create_dataset("timestamps", data=np.array(results.timestamps))
        res.create_dataset("function_calls", data=np.array(results.function_calls))
        counts_group = res.create_group("Counts")
        counts_group.create_dataset("states", data=np.array(counts_array_states))
        counts_group.create_dataset("counts", data=np.array(counts_array_vals))