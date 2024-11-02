"""
counts.py

Defines the `Counts` class and related functions for managing and analyzing measured quantum states.

This module provides tools for storing and processing measurement counts in quantum algorithms.
Key components include:

- **Counts class**: Stores states, counts, and energies. The class also calculates state probabilities and provides mappings for easy access.
- **Expectation Functions**: Includes `energy_expectation_value`, `filter_expectation_value`, and `filter_square_expectation_value` for computing expectation values with various filters.
- **State Probability Retrieval**: `get_state_prob` function retrieves the probability of a specific state based on the counts data.

"""


import numpy as np
from typing import Callable
from numpy.typing import NDArray
from .typing import FilterFunc

class Counts:
    """Class for storing measured states, the associated counts, and energies.

    The states are stored in a **dense bit representation**, where each state is represented as an integer.

    Args:
        states (NDArray[int]): Array of integer representations of measured states.
        counts (NDArray[int]): Array of counts corresponding to each state in `states`.
        energy_fn (Callable[[int], float]): A function that calculates the energy of a given state.

    Attributes:
        states (NDArray[int]): Array of measured states in dense bit representation.
        counts (NDArray[int]): Array of counts for each measured state.
        energies (NDArray[float]): 
            Array of energies corresponding to each state, calculated using `energy_fn`.
        num_counts (int): The total number of counts, calculated as the sum of `counts`.
        probs (NDArray[float]): Array of probabilities for each state.
        states_map (dict[int, int]): 
            A dictionary mapping each state to its index in the `states` array.
    """
    
    def __init__(self, states: NDArray[int], counts: NDArray[int], energy_fn: Callable[[int], float]):
        self.states = states
        self.counts = counts
        self.energies = np.array([energy_fn(s) for s in states])
        self.num_counts = counts.sum()
        self.probs = self.counts/self.num_counts
        self.states_map = {x: idx for idx,x in enumerate(states)}

def energy_expectation_value(counts: Counts) -> float:
    """Calculates the expectation value of energy based on state probabilities.

    Args:
        counts (Counts): An instance of the Counts class
            containing state probabilities and energies.

    Returns:
        float: The expectation value of the energy.
    """
    return (counts.probs * counts.energies).sum()

def filter_expectation_value(counts: Counts, filter_fn: FilterFunc, tau: float = 1.) -> float:
    """Calculates the expectation value fo the filtering operator corresponding to `filter_fn`.

    Args:
        counts (Counts): An instance of the Counts class containing 
            state probabilities and energies.
        filter_fn (FilterFunc): The filter function.
        tau (float, optional): A parameter for the filtering function. 
            Default is 1.

    Returns:
        float: The expectation value of the filtering operator.
    """
    return (counts.probs * filter_fn(counts.energies, tau)).sum()

def filter_square_expectation_value(counts: Counts, filter_fn: FilterFunc, tau: float = 1.) -> float:
    """Calculates the expectation value of the square filtering operator.

    Args:
        counts (Counts): An instance of the Counts class.
        filter_fn (FilterFunc): The filter function.
        tau (float, optional): A parameter for the filtering function. 
            Default is 1.

    Returns:
        float: The expectation value of the squared filtering operator.
    """
    return (counts.probs * filter_fn(counts.energies, tau)**2).sum()

def get_state_prob(counts: Counts, x: int) -> float:
    """
    Get the probability of observing the target state in the measurement counts.

    Args:
        counts (dict[str, int]): Measurement counts for different states.
        state (str): The target state.

    Returns:
        float: Probability of the target state in the measured results.
    """
    return counts.probs[counts.states_map[x]] if x in counts.states_map else 0.