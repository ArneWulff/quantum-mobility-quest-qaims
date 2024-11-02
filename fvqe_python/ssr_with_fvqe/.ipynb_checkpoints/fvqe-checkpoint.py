"""
fvqe.py

Implements the Filtering Variational Quantum Eigensolver (F-VQE) for combinatorial optimization.

This module provides an implementation of F-VQE, an algorithm designed to approximate ground states 
of problem Hamiltonians for combinatorial optimization tasks. F-VQE introduces a tunable filter 
parameter `tau` to control the optimization landscape and selectively reduce contributions from 
higher-energy states, enhancing convergence toward desired solutions.

F-VQE is based on the following paper:
    *Amaro, D., Modica, C., Rosenkranz, M., Fiorentini, M., Benedetti, M., & Lubasch, M. (2022).
    Filtering variational quantum algorithms for combinatorial optimization. Quantum Science and Technology, 7(1), 015021.*
    DOI: 10.1088/2058-9565/ac3e54.
    Available at: https://iopscience.iop.org/article/10.1088/2058-9565/ac3e54

Key components:
- **FvqeResults**: Class for storing results from the F-VQE optimization process, including energy values, 
  parameters, taus, gradients, and timestamps.
- **Gradient and Parameter Updates**: Functions to calculate gradients based on parameter shifts and update 
  parameters with a tunable learning rate.
- **Tau Adjustment**: Functions to find and adjust the optimal `tau` parameter for filtering energies and 
  controlling gradient norms during optimization.
- **Run F-VQE**: Main function `run_fvqe` to execute the F-VQE optimization loop, accommodating batch parameter 
  evaluations, adaptive tau increments, and handling interruptions.

This module supports various variational quantum circuit architectures (HWE, MERA, repeated MERA, and ply 
permutation circuits), enabling flexibility in circuit design. The `run_fvqe` function allows customization 
of optimization parameters, target states, and batching strategies, and includes handling for saving 
intermediate results if interrupted.
"""

import time
import numpy as np
from typing import Optional, Sequence, Callable
from numpy.typing import NDArray
from collections.abc import Sequence as SequenceABC

from .typing import (
    QubitState, QubitFunc, CountsDict, FilterFunc,
    CircuitParams, ParIndsFunc
)
from .helper_functions import timestr
from .encoding import state_to_dense, dense_to_state
from .counts import Counts, get_state_prob, energy_expectation_value, filter_expectation_value, filter_square_expectation_value
from .parameterized_state import (
    parameterized_hwe_state, parameterized_mera_state, parameterized_mera_rep_state
)
from .parameterized_state_perm import parameterized_perm_state

_FILTER_FUNCS: dict[str, FilterFunc] = {
    'inverse':     lambda x, t: (x + 0.001)**(-t),
    'logarithm':   lambda x, t: (-log(x+0.001))**t,
    'exponential': lambda x, t: np.exp(-t * x),
    'power':       lambda x, t: (1 - x)**t,
    'cosine':      lambda x, t: np.cos(x)**t,
    'linear':      lambda x, t: -t * x
}

def get_filter_fn(filter_fn: str):
    """Retrieves the filtering function based on a specified string key.

    Args:
        filter_fn (str): The name of the filter function to retrieve. 
            Possible values include:
            - 'inverse': Uses an inverse filter function, (x + 0.001)^(-t)
            - 'logarithm': Uses a logarithmic filter, (-log(x + 0.001))^t
            - 'exponential': Uses an exponential filter, exp(-t * x)
            - 'power': Uses a power filter, (1 - x)^t
            - 'cosine': Uses a cosine filter, cos(x)^t
            - 'linear': Uses a linear filter, -t * x

    Returns:
        FilterFunc: The specified filtering function, which accepts an energy value and
        tau parameter and returns the filtered energy.
    """
    return _FILTER_FUNCS[filter_fn]


def get_shifted_counts(
    parameters: CircuitParams,
    get_counts_fn: Callable[[CircuitParams], Counts | list[Counts]],
    par_inds: Optional[Sequence[int]] = None
) -> tuple[list[Counts], list[Counts]]:
    """Computes the shifted counts for the +π/2 and -π/2 parameter shifts.

    If `par_inds` is provided, only the specified subset of parameters will be shifted.

    Args:
        parameters (CircuitParams): Current set of circuit parameters.
        get_counts_fn (Callable[[CircuitParams], Counts | list[Counts]]): 
            Function to obtain counts for a given set of parameters, either singly or in batch.
        par_inds (Optional[Sequence[int]], optional): Sequence of indices specifying 
            which parameters to shift. If None, all parameters are shifted.

    Returns:
        tuple[list[Counts], list[Counts]]: A tuple containing two lists:
            - The first list has counts for parameters shifted by +π/2.
            - The second list has counts for parameters shifted by -π/2.
    """
    # If no subset is provided, use all parameters
    if par_inds is None:
        par_inds = np.arange(len(parameters))

    num_subset_params = len(par_inds)

    # Create arrays of shifted parameters for +π/2 and -π/2 shifts
    params_plus = np.tile(parameters, (num_subset_params, 1))
    params_minus = np.tile(parameters, (num_subset_params, 1))

    # Modify only the selected subset of parameters

    params_plus[np.arange(num_subset_params), par_inds] += np.pi/2
    params_minus[np.arange(num_subset_params), par_inds] -= np.pi/2

    # Stack the +π/2 and -π/2 shifted parameters together for batch processing
    all_shifted_params = np.vstack((params_plus, params_minus))

    # Get counts for all shifted parameters in a single call to get_counts_fn
    counts_list = get_counts_fn(all_shifted_params)

    # Split counts into +π/2 and -π/2 shifted counts
    num_params = len(params_plus)
    counts_shifted_plus = counts_list[:num_params]  # First half corresponds to +π/2 shifts
    counts_shifted_minus = counts_list[num_params:]  # Second half corresponds to -π/2 shifts

    return counts_shifted_plus, counts_shifted_minus

def calculate_gradient_component(
    counts: Counts,
    counts_plus: Counts,
    counts_minus: Counts,
    filter_fn: FilterFunc,
    tau: float
) -> float:
    """Calculates a single component of the gradient of the cost function using the parameter-shift rule.

    Args:
        counts (Counts): Counts for the current parameters.
        counts_plus (Counts): Counts for the parameters shifted by +π/2.
        counts_minus (Counts): Counts for the parameters shifted by -π/2.
        filter_fn (FilterFunc): Function for filtering energies based on tau.
        tau (float): Filtering parameter.

    Returns:
        float: The calculated gradient component for the parameter shift.
    """
    expectation_value_ft2 = filter_square_expectation_value(
        counts, filter_fn, tau
    )
    
    expectation_plus = filter_expectation_value(
        counts_plus, filter_fn, tau
    )
    expectation_minus = filter_expectation_value(
        counts_minus, filter_fn, tau
    )

    gradient = -(expectation_plus - expectation_minus) / (4 * expectation_value_ft2)
    return gradient

def calculate_full_gradient(
    parameters: CircuitParams,
    counts: Counts,
    counts_shifted_plus: Sequence[Counts],
    counts_shifted_minus: Sequence[Counts],
    filter_fn: FilterFunc,
    tau: float | Sequence[float],
    par_inds: Optional[Sequence[int]] = None
) -> NDArray:
    """Calculates the full gradient vector for the cost function using the parameter-shift rule.

    Args:
        parameters (CircuitParams): Current set of variational parameters.
        counts (Counts): Counts for the current parameters.
        counts_shifted_plus (Sequence[Counts]): List of counts for the +π/2 shifted parameters.
        counts_shifted_minus (Sequence[Counts]): List of counts for the -π/2 shifted parameters.
        filter_fn (FilterFunc): Function for filtering energies based on tau.
        tau (float | Sequence[float]): Filtering parameter, either a single value or a sequence 
            if parameters have different taus.
        par_inds (Optional[Sequence[int]], optional): Sequence of indices specifying which 
            parameters to include in the gradient. Defaults to None.

    Returns:
        NDArray: The calculated gradient vector.
    """
    if par_inds is None:
        par_inds = np.arange(len(parameters))

    if isinstance(tau, SequenceABC):
        assert len(tau) == len(par_inds)
        tau_seq = tau
    else:
        tau_seq = np.full(len(par_inds), tau)
    
    
    gradients = np.zeros_like(parameters)

    for (j, t, counts_plus, counts_minus) in zip(
        par_inds, tau_seq, counts_shifted_plus, counts_shifted_minus
    ):
        gradients[j] = calculate_gradient_component(
            counts, counts_plus, counts_minus, filter_fn, t
        )

    return gradients

def parameter_update(
    parameters: CircuitParams,
    gradient: NDArray[float],
    learning_rate: float,
    par_inds: Optional[Sequence[int]] = None
) -> CircuitParams:
    """Updates the parameters based on the gradient and learning rate.

    Args:
        parameters (CircuitParams): Current set of circuit parameters as a 1d numpy array.
        gradient (NDArray[float]): Gradient of the loss function with respect to the parameters.
        learning_rate (float): Learning rate for the parameter update.
        par_inds (Optional[Sequence[int]], optional): Indices of parameters to update. If None, all parameters are updated.

    Returns:
        CircuitParams: Updated parameters after applying the gradient descent step.
    """
    if par_inds is not None:
        mask = np.zeros_like(gradient)
        mask[par_inds] = 1
        gradient = gradient*mask
    return parameters - learning_rate * gradient

def binary_search_tau(
    parameters: CircuitParams,
    tau_low: float,
    tau_high: float,
    g_c: float,
    filter_fn: FilterFunc,
    counts: Counts,
    counts_shifted_plus: list[Counts],
    counts_shifted_minus: list[Counts],
    precision_tau: float, precision_gradient: float,
    par_inds: Optional[Sequence[int]] = None
) -> float:
    """Performs a binary search to find the optimal tau within a specified range.

    Args:
        parameters (CircuitParams): Current variational parameters.
        tau_low (float): Lower bound for tau.
        tau_high (float): Upper bound for tau.
        g_c (float): Target gradient norm.
        filter_fn (FilterFunc): Function for filtering energies based on tau.
        counts (Counts): Counts for the current parameters.
        counts_shifted_plus (list[Counts]): List of counts for parameters shifted by +π/2.
        counts_shifted_minus (list[Counts]): List of counts for parameters shifted by -π/2.
        precision_tau (float): Precision for tau in the binary search.
        precision_gradient (float): Precision for the gradient norm.
        par_inds (Optional[Sequence[int]], optional): Indices of parameters to optimize.

    Returns:
        float: The optimal tau value within the specified precision.
    """
    while tau_high - tau_low > precision_tau:
        tau_mid = (tau_high + tau_low) / 2.0

        # Calculate the full gradient for the current tau_mid
        gradients = calculate_full_gradient(
            parameters, counts, counts_shifted_plus, counts_shifted_minus, 
            filter_fn, tau_mid,
            par_inds=par_inds
        )
        gradient_norm = np.linalg.norm(gradients)

        if gradient_norm > g_c:
            tau_high = tau_mid
        else:
            tau_low = tau_mid

        if abs(gradient_norm - g_c) < precision_gradient:
            return tau_mid

    return tau_low

def find_tau(
    parameters: CircuitParams,
    filter_fn: FilterFunc,
    counts: Counts,
    counts_shifted_plus: list[Counts],
    counts_shifted_minus: list[Counts],
    g_c: float,
    tau_increment: float = 0.1,
    max_tau: float = 10.0,
    precision_tau: float = 1e-2, precision_gradient: float = 1e-2,
    par_inds: Optional[Sequence[int]] = None
) -> float:
    """Finds the optimal tau that keeps the gradient norm close to a target value.

    Args:
        parameters (CircuitParams): Current variational parameters.
        filter_fn (FilterFunc): Function for filtering energies based on tau.
        counts (Counts): Counts for the current parameters.
        counts_shifted_plus (list[Counts]): List of counts for parameters shifted by +π/2.
        counts_shifted_minus (list[Counts]): List of counts for parameters shifted by -π/2.
        g_c (float): Target gradient norm.
        tau_increment (float, optional): Increment for tau search. Defaults to 0.1.
        max_tau (float, optional): Maximum tau to consider. Defaults to 10.0.
        precision_tau (float, optional): Precision for tau in the binary search. Defaults to 1e-2.
        precision_gradient (float, optional): Precision for the gradient norm. Defaults to 1e-2.
        par_inds (Optional[Sequence[int]], optional): Indices of parameters to optimize.

    Returns:
        float: The optimal tau value.
    """
    tau = tau_increment
    best_tau = tau
    best_gradient_norm = 0.0

    while tau <= max_tau:
        # Calculate the full gradient vector and its norm
        gradients = calculate_full_gradient(
            parameters, counts, counts_shifted_plus, counts_shifted_minus, filter_fn, tau,
            par_inds=par_inds
        )
        gradient_norm = np.linalg.norm(gradients)

        if gradient_norm > g_c:
            return binary_search_tau(
                parameters, best_tau, tau, g_c, 
                filter_fn, counts, 
                counts_shifted_plus, counts_shifted_minus,
                precision_tau, precision_gradient, par_inds=par_inds
            )

        if gradient_norm > best_gradient_norm and gradient_norm < g_c:
            best_gradient_norm = gradient_norm
            best_tau = tau

        tau += tau_increment

    return best_tau

class FvqeResults:
    """Stores results from the F-VQE optimization process.

    Attributes:
        energies (NDArray): Array of energy values over iterations.
        parameters (NDArray): Array of parameter values over iterations.
        taus (NDArray): Array of tau values over iterations.
        gradients (NDArray): Array of gradient values over iterations.
        timestamps (NDArray): Array of timestamps for each iteration.
        counts (list[CountsDict]): List of measurement counts for each iteration.
        function_calls (NDArray): Number of function calls for each iteration.
        target_probs (Optional[NDArray]): Array of probabilities for reaching the target state, if specified.
    """
    def __init__(self, energies: NDArray, parameters: NDArray,
                 taus: NDArray, gradients: NDArray, timestamps: NDArray,
                 counts: list[CountsDict], function_calls: NDArray,
                 target_probs: Optional[NDArray] = None):
        self.energies = energies
        self.parameters = parameters
        self.taus = taus
        self.gradients = gradients
        self.timestamps = timestamps
        self.counts = counts
        self.function_calls = function_calls
        self.target_probs = target_probs


def run_fvqe(
    num_qubits: int,
    num_reps: int,
    energy_fn: QubitFunc,
    filter_fn: FilterFunc,
    x0: Optional[CircuitParams] = None,
    shots: int = 1000,
    maxiter: int = 100,
    learning_rate: float = 1.,
    max_tau: float = 10.,
    tau_increment: float = 0.1,
    adapt_tau_increment: bool = False,
    par_inds_fn: Optional[ParIndsFunc] = None,
    reverse_par_inds: bool = False,
    g_c: float = 0.1,
    renormalize_gc: bool = False,
    target_state: str = None,
    previous_results: Optional[FvqeResults] = None,
    vqc: str = 'hwe',
    print_info: bool = False,
    perm_initial_state: Optional[str] = None
) -> FvqeResults:
    """Runs the F-VQE optimization algorithm for combinatorial optimization.

    The Filtering Variational Quantum Eigensolver (F-VQE) is an algorithm designed to tackle 
    combinatorial optimization problems by approximating the ground state of a problem Hamiltonian.
    This implementation follows the approach described in the paper:
    
        *Amaro, D., Modica, C., Rosenkranz, M., Fiorentini, M., Benedetti, M., & Lubasch, M. (2022).
        Filtering variational quantum algorithms for combinatorial optimization. Quantum Science 
        and Technology, 7(1), 015021.* DOI: 10.1088/2058-9565/ac3e54.
        Available at: https://iopscience.iop.org/article/10.1088/2058-9565/ac3e54

    The F-VQE algorithm iteratively updates variational parameters using a filtered expectation 
    of energy, which is controlled by a tunable parameter `tau`. The algorithm seeks to balance 
    the gradient norm with a target value `g_c`, adjusting `tau` as needed to control the optimization 
    landscape. This function also handles parameter batching, adaptive `tau` increments, and 
    keyboard interrupts to save progress if interrupted.

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        num_reps (int): Number of repetitions of the variational circuit layers.
        energy_fn (QubitFunc): Function to calculate the energy of each sampled state.
        filter_fn (FilterFunc): Function for filtering energies based on tau.
        x0 (Optional[CircuitParams], optional): Initial parameters. Defaults to None.
        shots (int, optional): Number of samples per measurement. Defaults to 1000.
        maxiter (int, optional): Maximum number of optimization iterations. Defaults to 100.
        learning_rate (float, optional): Learning rate for parameter updates. Defaults to 1.0.
        max_tau (float, optional): Maximum tau to consider in the optimization. Defaults to 10.0.
        tau_increment (float, optional): Increment for tau search. Defaults to 0.1.
        adapt_tau_increment (bool, optional): Whether to adapt tau increment during optimization.
            Defaults to False.
        par_inds_fn (Optional[ParIndsFunc], optional): Function to generate parameter indices for
            each iteration. Defaults to None.
        reverse_par_inds (bool, optional): Whether to reverse parameter indices. Defaults to False.
        g_c (float, optional): Target gradient norm. Defaults to 0.1.
        renormalize_gc (bool, optional): Whether to renormalize the target gradient norm. Defaults to False.
        target_state (str, optional): Target state in binary format. Defaults to None.
        previous_results (Optional[FvqeResults], optional): Previous results to resume optimization from
            Defaults to None.
        vqc (str, optional): Type of variational quantum circuit ('hwe', 'mera', 'merarep', or 'perm').
            Defaults to 'hwe'.
        print_info (bool, optional): Whether to print information during optimization. Defaults to False.
        perm_initial_state (Optional[str], optional): Initial state for 'perm' type circuit. Defaults to None.

    Returns:
        FvqeResults: The results of the F-VQE optimization process, including energies, parameters,
            taus, gradients, and timestamps.
        
    Notes:
        - Keyboard interrupts (e.g., Ctrl+C) will save the results up to the current iteration, 
          allowing you to resume from the last saved state if the optimization is interrupted.
        - The `tau` parameter is optimized through a binary search to ensure that the gradient 
          norm stays close to the target value `g_c`, making the algorithm adaptive to the
          optimization landscape.
        - The F-VQE algorithm supports multiple variational quantum circuit types, specified by `vqc`.
    """
    match vqc:
        case 'hwe':
            qc_state = parameterized_hwe_state(num_qubits, num_reps)
        case 'mera':
            qc_state = parameterized_mera_state(num_qubits)
        case 'merarep':
            qc_state = parameterized_mera_rep_state(num_qubits)
        case 'perm':
            if perm_initial_state is None:
                ValueError(f"for vqc='perm', initial state must be provided.")
            qc_state = parameterized_perm_state(perm_initial_state,num_reps)
        case _:
            raise ValueError(f"`vqc` must be `'hwe'` or `'mera'`, not {vqc}")
    num_pars = qc_state.num_params

    # If previous results are provided, resume from the last parameters and counts.
    if previous_results is not None:
        counts_hist = list(previous_results.counts)
        energies = list(previous_results.energies)
        parameters = list(previous_results.parameters)
        taus = list(previous_results.taus)
        gradients = list(previous_results.gradients)
        timestamps = list(previous_results.timestamps)
        function_calls = list(previous_results.function_calls)
        target_probs = (
            list(previous_results.target_probs) 
            if previous_results.target_probs is not None else None
        )
        # Shift new timestamps based on the last timestamp in previous results
        time_offset = previous_results.timestamps[-1]
    else:
        counts_hist = []
        energies = []
        parameters = []
        taus = []
        gradients = []
        timestamps = []
        function_calls = []
        target_probs = [] if target_state is not None else None
        time_offset = 0

    if target_state is not None:
        target_state_dense = state_to_dense(target_state)
    
    iter_offset = len(timestamps)

    # If no par_inds_fn is provided, use all parameters (None means all).
    if par_inds_fn is None:
        par_inds_fn = lambda _: None

    objective_function_calls = 0

    # Function to handle batch processing or individual calls to get measurement counts.
    def get_counts_fn(x: CircuitParams) -> Counts:
        nonlocal objective_function_calls
        if x.ndim > 2:
            raise ValueError(
                f"Dimension of x must be 1 or 2 (for batches), not x.ndim = {x.ndim}"
            )
        if x.ndim == 2:
            objective_function_calls += x.shape[0]
            return get_counts_fn_batch(x)
        objective_function_calls += 1
        return qc_state.sample(x, energy_fn, shots=shots)

    # Batch processing function for parameter sets.
    def get_counts_fn_batch(x_array: CircuitParams) -> list[Counts]:
        """Handles batches of parameter sets."""
        return [qc_state.sample(x, energy_fn, shots=shots) for x in x_array]

    if previous_results is None:
        # Initialize starting parameters if not resuming from previous results
        # and x0 not specified.
        if x0 is None:
            if vqc == 'perm':
                x0 = np.full(num_pars, np.pi/2, dtype=float)
            else:
                x0 = np.zeros(num_pars, dtype=float)
                x0[qc_state.get_last_parameter_indices()] = np.pi / 2

        if print_info:
            print("Starting optimization...")

        x0_counts = get_counts_fn(x0)
        x0_energy = energy_expectation_value(x0_counts)
        if target_state is not None:
            x0_target_prob = get_state_prob(x0_counts, target_state_dense)

        parameters.append(x0)
        energies.append(x0_energy)
        counts_hist.append({dense_to_state(s, num_qubits): c for s,c in zip(x0_counts.states,x0_counts.counts)})
        if target_state is not None:
            target_probs.append(x0_target_prob)

        last_counts = x0_counts
    else:
        last_states, last_counts = zip(*counts_hist[-1].items())
        last_states = [state_to_dense(s) for s in last_states]
        last_counts = Counts(
            np.array(last_states),np.array(last_counts),energy_fn,num_qubits
        )
        
    if print_info:
        print(f"Initial energy: {energies[-1]:.4f}")
        if target_state is not None:
            print(f"Initial target state probabilty: {target_probs[-1]:.4f}")
    print()

    t0 = time.time()

    # Run F-VQE for maxiter iterations
    for itr in range(iter_offset, maxiter+iter_offset):
        try:
            if print_info:
                print(f"Iteration {itr}")
                print("-" * (len(f"Iteration {itr}")) + "\n")

            objective_function_calls = 0

            # Use the last result's parameters and counts or the initial ones.
            pars = parameters[-1]
            counts = last_counts

            # Get the parameter indices for this iteration
            par_inds_list = par_inds_fn(itr)

            # wrap in list if necessary
            if par_inds_list is None or isinstance(par_inds_list[0], int):
                all_par_inds = (
                    list(par_inds_list) 
                    if par_inds_list is not None
                    else None
                )
                par_inds_list = [par_inds_list]
            else:
                assert isinstance(par_inds_list[0], SequenceABC)
                all_par_inds = list(set().union(*par_inds_list))
                assert len(all_par_inds) == sum(len(pinds) for pinds in par_inds_list)

            counts_plus, counts_minus = get_shifted_counts(
                pars, get_counts_fn, all_par_inds
            )

            # Reverse the parameter indices if specified
            for j in range(len(par_inds_list)):
                if reverse_par_inds and par_inds_list[j] is not None:
                    par_inds_list[j] = [
                        num_qubits - n - 1 for n in par_inds_list[j]
                    ]

            g_c_eff = (
                g_c / np.sqrt(len(par_inds_list)) 
                if renormalize_gc else g_c
            )

            # Tau search for all subsets
            tau_list = [find_tau(
                pars, filter_fn, counts, counts_plus, counts_minus, 
                g_c=g_c_eff, max_tau=max_tau, tau_increment=tau_increment, 
                par_inds=par_inds
            ) for par_inds in par_inds_list]

            tau_array = np.zeros(len(pars))
            for tau, par_inds in zip(tau_list, par_inds_list):
                if par_inds is None:
                    tau_array[:] = tau
                else:
                    tau_array[par_inds] = tau

            if print_info:
                print(f"average tau: {tau_array[all_par_inds].mean()}")

            gradient = calculate_full_gradient(
                pars, counts, counts_plus, counts_minus, 
                filter_fn, tau_array[all_par_inds], par_inds=all_par_inds
            )

            if print_info:
                print(f"gradient norm: {np.linalg.norm(gradient)}")

            updated_pars = parameter_update(
                pars, gradient, learning_rate, par_inds=all_par_inds
            )

            updated_counts = get_counts_fn(updated_pars)

            taus.append(tau_array)
            counts_hist.append({
                dense_to_state(s, num_qubits): c 
                for s,c in zip(updated_counts.states, updated_counts.counts)
            })
            parameters.append(updated_pars)
            energies.append(energy_expectation_value(updated_counts))
            function_calls.append(objective_function_calls)
            gradients.append(gradient)
            timestamps.append(time.time() - t0 + time_offset)
            if target_state is not None:
                target_probs.append(
                    get_state_prob(updated_counts, target_state_dense)
                )

            if adapt_tau_increment:
                tau_increment = tau_array[all_par_inds].mean() / 10

            last_counts = updated_counts
            
            if print_info:
                print(f"energy: {energies[-1]:.6f}")
                print(f"function calls: {objective_function_calls}")
                if target_state is not None:
                    print(f"target probability: {target_probs[-1]}")
                print()
                print(f"timestamp: {timestr(timestamps[-1])}")
                print()
            # print("\n-._.-*'*-._.-*'*-._.-*'*-._.-\n")

        except KeyboardInterrupt:
            print("Optimization interrupted. Saving results so far...")
            # remove potentially uncomlete data
            taus = taus[:itr]
            counts_hist = counts_hist[:itr]
            energies = energies[:itr+1]  # includes x0
            function_calls = function_calls[:itr]
            gradients = gradients[:itr]
            timestamps = timestamps[:itr]
            parameters = parameters[:itr+1]  # includes x0
            if target_state is not None:
                target_probs = target_probs[:itr+1]  # includes x0
            break

    
    return FvqeResults(
        np.array(energies),
        parameters,
        taus,
        gradients,
        np.array(timestamps),
        counts_hist,
        np.array(function_calls),
        np.array(target_probs) if target_state is not None else None
    )