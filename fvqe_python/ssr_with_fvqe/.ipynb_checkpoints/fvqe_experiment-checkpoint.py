"""
fvqe_buckling.py

This module provides functionality for running an F-VQE-based optimization experiment for stacking 
sequence retrieval in laminate design. It includes functions for setting up and managing F-VQE 
experiments with specific configurations, constraints, and encodings, as well as handling input/output 
for experiment results.

Classes:
    - `FvqeOptions`: Configuration options for running an F-VQE experiment, including 
      tau adjustments, learning rates, and variational circuit repetitions.
    
Functions:
    - `hdf5_to_fvqe_result`: Loads previously saved F-VQE results from an HDF5 file and 
      returns an `FvqeResults` instance with the loaded data.
    
    - `fvqe_experiment`: Sets up and executes an F-VQE experiment for optimizing stacking 
      sequences to meet specific laminate design criteria. It allows optional constraints 
      and multiple encoding schemes, providing flexibility for various laminate design 
      optimization tasks.

Usage:
    This module is inteded to be used in experiments for laminate design where stacking sequences 
    must be optimized to match target parameters using variational quantum algorithms. 
    Experiments can be configured using the `FvqeOptions` class, which includes settings 
    for learning rate, tau increments, target gradient norms, and other parameters relevant 
    to F-VQE optimization.

    Results of the F-VQE experiments, such as energies, gradients, taus, and function calls, 
    are saved to an HDF5 file for later analysis and can be reloaded using the `hdf5_to_fvqe_result` 
    function.
"""


import numpy as np
from typing import Optional, Callable, Sequence
from numpy.typing import NDArray
from os.path import isfile
import h5py
from .typing import Encoding, FilterFunc, CircuitParams, Stack, Parameters
from .laminate import Laminate
from .encoding import stack_to_state, convert_func_qd_to_qb, state_to_stack, convert_func_to_dense, dense_to_stack, state_to_dense
from .constraints import ConstraintSettings
from .fvqe import FvqeResults, run_fvqe, get_filter_fn

_DEFAULT_FILTER_FUNC = 'exponential'

def hdf5_to_fvqe_result(filename: str) -> FvqeResults:
    """Loads F-VQE results from an HDF5 file into an FvqeResults instance.

    This function reads data from an HDF5 file and returns an FvqeResults instance 
    with the stored results. It supports the standard HDF5 structure defined by 
    `fvqe_experiment` for loading various datasets, including counts and target 
    probabilities if available.

    Args:
        filename (str): Path to the HDF5 file containing F-VQE results.

    Returns:
        FvqeResults: An instance containing the loaded energies, parameters, taus, 
        gradients, timestamps, function calls, counts, and target probabilities (if present).
    """
    results_dict = {}
    with h5py.File(filename, 'r') as file:
        for ds in (
            'energies',
            'parameters',
            'taus',
            'gradients',
            'timestamps',
            'function_calls',            
        ):
            results_dict[ds] = np.array(file[f"Results/{ds}"])
        results_dict['target_probs'] = (
            np.array(file[f"Results/target_probs"])
            if 'target_probs' in file["Results"]
            else None
        )
        states = np.array(file[f"Results/Counts/states"])
        counts = np.array(file[f"Results/Counts/counts"])
    results_dict['counts'] = [
        {
            s.decode('utf-8'): int(c) 
            for s,c in zip(*entry) 
            if len(s) > 0
        }
        for entry in zip(states, counts)
    ]
    return FvqeResults(**results_dict)

class FvqeOptions:
    """Options for configuring the F-VQE algorithm during optimization.

    The `FvqeOptions` class consolidates various configuration options for the 
    `run_fvqe` function, allowing customization of parameters, learning rate, 
    tau adjustments, and other settings used to control the optimization.

    Attributes:
        num_reps (int): Number of repetitions of the variational quantum circuit layers.
        filter_fn (FilterFunc): Function for filtering energies based on tau.
        filter_fn_name (str): Name of the filter function.
        x0 (Optional[CircuitParams]): Initial parameters for the F-VQE run.
        shots (int): Number of samples per measurement.
        maxiter (int): Maximum number of optimization iterations.
        learning_rate (float): Learning rate for parameter updates.
        max_tau (float): Maximum tau value to consider during optimization.
        tau_increment (float): Increment for adjusting tau in optimization.
        adapt_tau_increment (bool): Whether to adapt tau increment during the optimization process.
        par_inds_fn (Optional[Callable[[int], None | Sequence[int] | Sequence[Sequence[int]]]]): 
            Function to generate parameter indices for each iteration.
        reverse_par_inds (bool): Whether to reverse parameter indices.
        g_c (float): Target gradient norm.
        renormalize_gc (bool): Whether to renormalize the target gradient norm.
    """
    def __init__(
        self,
        num_reps: int,
        filter_fn: Optional[str | FilterFunc] = None,
        filter_fn_name: Optional[str] = None,
        x0: Optional[CircuitParams] = None,
        shots: int = 1000,
        maxiter: int = 100,
        learning_rate: float = 1., 
        max_tau: float = 10.,
        tau_increment: float = 0.1,
        adapt_tau_increment: bool = False,
        par_inds_fn: Optional[Callable[[int], None | Sequence[int] | Sequence[Sequence[int]]]] = None,
        reverse_par_inds: bool = False,
        g_c: float = 0.1,
        renormalize_gc: bool = False,
    ):
        self.num_reps = num_reps

        if isinstance(filter_fn, str):
            self.filter_fn = get_filter_fn(filter_fn)
            filter_fn_name = filter_fn if filter_fn_name is None else filter_fn_name
        elif filter_fn is None:
            self.filter_fn = get_filter_fn(_DEFAULT_FILTER_FUNC)
            filter_fn_name = _DEFAULT_FILTER_FUNC if filter_fn_name is None else filter_fn_name
        else:
            self.filter_fn = filter_fn
            filter_fn_name = filter_fn.__name__ if filter_fn_name is None else filter_fn_name
        self.filter_fn_name = filter_fn_name

        self.x0 = x0
        self.shots = shots
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.max_tau = max_tau
        self.tau_increment = tau_increment
        self.adapt_tau_increment = adapt_tau_increment
        self.par_inds_fn = par_inds_fn
        self.reverse_par_inds = reverse_par_inds
        self.g_c = g_c
        self.renormalize_gc = renormalize_gc


def fvqe_experiment(
    filepath: str,
    laminate: Laminate,
    target_parameters: Parameters,
    encoding: Encoding,
    fvqe_options: FvqeOptions,
    target_stack: Optional[Stack] = None,
    lp_loss: str = 'rmse',
    constraint_settings: Optional[ConstraintSettings] = None,
    vqc: str = 'hwe',
    print_info: bool = False,
    resume_filepath: Optional[str] = None
):
    """Runs an F-VQE experiment for stacking sequence retrieval in laminate design.

    This function configures and runs an F-VQE optimization tailored to the 
    stacking sequence retrieval problem in laminate design. It defines a loss 
    function based on laminate parameters and, optionally, constraints on the 
    stacking sequence, then uses F-VQE to optimize the sequence. The results 
    are saved to an HDF5 file, which includes energy data, parameter values, 
    tau settings, and other relevant information.

    Args:
        filepath (str): Path to the file where experiment results will be saved.
            Must be a valid path for an HDF5 file (e.g., ends with '.hdf5').
        laminate (Laminate): Instance containing laminate parameters and properties.
        target_parameters (Parameters): Target parameters for the laminate design.
        encoding (Encoding): Encoding format for the stacking sequence.
        fvqe_options (FvqeOptions): Configurations and options for running F-VQE.
        target_stack (Optional[Stack], optional): Target stacking sequence for 
            validation purposes. Defaults to None.
        lp_loss (str, optional): Loss function type ('rmse' or 'mse'). Defaults to 'rmse'.
        constraint_settings (Optional[ConstraintSettings], optional): Constraints on the 
            stacking sequence to be applied during optimization. Defaults to None.
        vqc (str, optional): Type of variational quantum circuit to use ('hwe', 'mera', 
            'merarep', or 'perm'). Defaults to 'hwe'.
        print_info (bool, optional): Whether to print progress information during the run.
            Defaults to False.
        resume_filepath (Optional[str], optional): Path to an existing results file to 
            resume an experiment. Defaults to None.

    Notes:
        - The function supports saving target probabilities if a `target_stack` is provided,
          making it suitable for testing the optimization performance against known results.
        - The resulting HDF5 file has a fixed structure for Properties, Laminate settings, 
          Constraints, Encoding, and F-VQE configurations.
    """
    if isfile(filepath):
        raise ValueError(f"File {filepath} already exists.")

    num_qubits = laminate.num_plies * len(encoding[0])

    if lp_loss == 'rmse':
        loss_fn = convert_func_to_dense(
            lambda x: np.sqrt(
                np.sum((laminate.parameters(x) - target_parameters)**2)
            ),
            num_qubits,
            encoding
        )
    elif lp_loss == 'mse':
        loss_fn = convert_func_to_dense(
            lambda x: np.sum((laminate.parameters(x) - target_parameters)**2),
            num_qubits,
            encoding
        )
    else:
        raise ValueError(f"`lp_loss` must be `'rmse'` or `'mse'`, not `'{lp_loss}'`")

    if constraint_settings is None:
        energy_fn = loss_fn
    else:
        energy_fn = lambda x: loss_fn(x) + constraint_settings.penalty(dense_to_stack(x, num_qubits, encoding))

    if target_stack is not None:
        target_state = stack_to_state(target_stack, encoding)
    else:
        target_state = None

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
        target_state=target_state,
        vqc=vqc,
        previous_results=previous_results,
        print_info=print_info
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
        if target_stack is not None:
            props.create_dataset("target_stack", data=np.array(target_stack))
            props.attrs["target_state"] = target_state

        props_lam = props.create_group("Laminate")
        props_lam.attrs["num_plies"] = laminate.num_plies
        props_lam.attrs["num_angles"] = laminate.num_angles
        props_lam.attrs["num_weights"] = laminate.num_weights
        props_lam.attrs["num_funcs"] = laminate.num_funcs

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
        props_q.attrs["lp_loss"] = lp_loss

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
        if target_state is not None:
            res.create_dataset("target_probs", data=np.array(results.target_probs))
        counts_group = res.create_group("Counts")
        counts_group.create_dataset("states", data=np.array(counts_array_states))
        counts_group.create_dataset("counts", data=np.array(counts_array_vals))