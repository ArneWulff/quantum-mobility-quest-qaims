"""
laminate.py

Defines tools for configuring and calculating lamination parameters of composite laminates.

This module provides functionality for setting up the weights and function arrays needed
to characterize the in-plane, bending, and coupling properties of a laminate using lamination
parameters. The main class, `Laminate`, represents a laminate with specified ply orientations
and allows calculation of parameters for various stacking sequences.

Classes:
    Laminate: Defines the properties of a laminate, supporting the calculation of lamination parameters.

Primary Functions:
    generate_weights: Calculates weights for different types of lamination parameters (A, B, D).
    generate_funcs: Generates function values for allowed ply angles.
    create_laminate: Creates a Laminate instance with specified plies and angle functions.
"""

from typing import Optional, Sequence, Collection
import numpy as np
from numpy.typing import NDArray

from .typing import Stack, WeightsArray, FuncArray, AngleFunction, Parameters

_DEFAULT_ANGLE_FUNCTIONS: tuple[AngleFunction, ...] = (
    lambda x: np.cos(2 * x),
    lambda x: np.sin(2 * x),
    lambda x: np.cos(4 * x),
    lambda x: np.sin(4 * x),
)

def generate_weights_a(num_plies: int, symmetric: bool = False) -> NDArray[float]:
    """Calculates the weights for the A lamination parameters

    Args:
        num_plies (int): number of plies
        symmetric (bool): whether the laminate is symmetric.
            Here, this has no effect.

    Returns:
        NDArray[float]: the weights, of shape (num_plies,)
    """
    return np.full(num_plies, 1 / num_plies)


def generate_weights_b(num_plies: int, symmetric: bool = False) -> NDArray[float]:
    """Calculates the weights for the B lamination parameters

    Args:
        num_plies (int): number of plies
        symmetric (bool): whether the laminate is symmetric.
            Here `True` raises an error.

    Returns:
        NDArray[float]: the weights, of shape `(num_plies,)`
    """
    if symmetric:
        raise ValueError("B-parameters only available for non-symmetric laminates.")
    boundaries_b = 2 * (np.arange(0, num_plies + 1) / num_plies - 1 / 2) ** 2
    return boundaries_b[1:] - boundaries_b[:-1]


def generate_weights_d(num_plies: int, symmetric: bool = False) -> NDArray[float]:
    """Calculates the weights for the D lamination parameters

    Args:
        num_plies (int): number of plies
        symmetric (bool): whether the laminate is symmetric.

    Returns:
        NDArray[float]: the weights, of shape `(num_plies,)`
    """
    if symmetric:
        boundaries_d = (np.arange(0, num_plies + 1) / num_plies) ** 3
    else:
        boundaries_d = 4 * (np.arange(0, num_plies + 1) / num_plies - 1 / 2) ** 3
    return boundaries_d[1:] - boundaries_d[:-1]


_WEIGHT_GENERATORS = {
    'A': generate_weights_a,
    'B': generate_weights_b,
    'D': generate_weights_d
}


def generate_weights(num_plies: int, symmetric: bool = True, which: bool = None) -> WeightsArray:
    """
    Calculates weights for lamination parameters (A, B, D) based on the number of plies and symmetry.

    For symmetric laminates, the weights are indexed from the midplane outward. The weights array
    is used to calculate the lamination parameters for different stacking sequences.

    Args:
        num_plies (int): Number of plies (or half for symmetric laminates).
        symmetric (bool, optional): Whether the laminate is symmetric. Defaults to True.
        which (str, optional): Specifies which lamination parameters ('A', 'B', 'D') to generate.
            Defaults to 'AD' if symmetric, otherwise 'ABD'.

    Returns:
        WeightsArray: A numpy array of shape `(len(which), num_plies)` containing weights for the
            specified lamination parameters.
    """
    if which is None:
        which = 'AD' if symmetric else 'ABD'
    which = which.upper()
    if symmetric and 'B' in which:
        raise ValueError("B-parameters only available for non-symmetric laminates.")

    return np.stack([
        _WEIGHT_GENERATORS[w](num_plies, symmetric) for w in which
    ])


def generate_funcs(
        angles: Sequence[int | float], angle_functions: Optional[Sequence[AngleFunction]] = None,
        deg: bool = False, round_decimals: Optional[int] = None
) -> FuncArray:
    """
    Generates function values for each allowed ply angle based on the provided angle functions.

    Each angle function evaluates a ply angle and returns a function value used in the calculation
    of lamination parameters. By default, the functions are [cos(2x), sin(2x), cos(4x), sin(4x)].

    Args:
        angles (Sequence[int | float]): Allowed ply angles.
        angle_functions (Sequence[AngleFunction], optional): Ply angle functions for lamination
            parameter calculations. If None, defaults to [cos(2x), sin(2x), cos(4x), sin(4x)].
        deg (bool, optional): Indicates if angles are in degrees. Defaults to False.
        round_decimals (int, optional): Rounds function values to a specified number of decimals.
            If None, no rounding is applied.

    Returns:
        FuncArray: Array of shape `(len(angles), len(angle_functions))` containing function values
            for each ply angle.
    """
    if angle_functions is None:
        angle_functions = _DEFAULT_ANGLE_FUNCTIONS

    funcs = np.array([
        [f(a * np.pi / 180 if deg else a) for f in angle_functions]
        for a in angles
    ])

    return funcs if round_decimals is None else funcs.round(decimals=round_decimals)

        

class Laminate:
    """
    Represents a laminate with ply-dependent weights and ply-angle functions.

    The `Laminate` class provides an interface for calculating lamination parameters
    based on the weights and functions arrays. These parameters characterize the
    mechanical properties of a composite laminate and are essential for stiffness
    and buckling calculations.

    Note:
        The laminate's properties depend on the input weights and functions arrays.
        The allowed ply angles are implicitly included through the functions array,
        and not stored as a separate attribute.

    Args:
        weights (WeightsArray): Array of shape `(num_weights, num_plies)` containing the
            weights for lamination parameters (typically A, B, D).
        funcs (FuncArray): Array of shape `(num_angles, num_funcs)` containing function values
            for allowed ply angles, with each column representing a different ply-angle function.

    Attributes:
        weights (WeightsArray): The weights array representing lamination parameter weights.
        funcs (FuncArray): The functions array with ply-angle function evaluations.
        num_plies (int): Number of plies in the laminate.
        num_angles (int): Number of ply angles considered.
        num_weights (int): Number of distinct sets of weights (typically A, B, D).
        num_funcs (int): Number of ply-angle functions used.
        num_parameters (int): Total number of lamination parameters, equal to
            `num_weights * num_funcs`.

    Methods:
        parameters: Calculates lamination parameters for a given stacking sequence.
    """

    def __init__(self, weights: WeightsArray, funcs: FuncArray):
        self.weights: WeightsArray = weights
        self.funcs: FuncArray = funcs

        self.num_plies: int
        self.num_angles: int
        self.num_weights: int
        self.num_funcs: int
        self.num_weights, self.num_plies = self.weights.shape
        self.num_angles, self.num_funcs = self.funcs.shape
        self.num_parameters: int = self.num_weights * self.num_funcs

    def parameters(self, stack: Stack) -> Parameters:
        """
        Calculate lamination parameters for a given stacking sequence.

        Args:
            stack (Stack): Stacking sequence as a numpy array of integers, where each integer
                corresponds to a ply angle index in `num_angles`.

        Returns:
            Parameters: Numpy array of shape `(num_weights, num_funcs)` containing the
                calculated lamination parameters for the specified stacking sequence.
        """
        return self.weights @ self.funcs[stack]

def create_laminate(
    num_plies: int, angles: Sequence[int | float],
    symmetric: bool = False, weight_types: Optional[str] = None,
    angle_functions: Optional[Sequence[AngleFunction]] = None,
    deg: bool = False, funcs_round_decimals: Optional[int] = None
) -> Laminate:
    """
    Creates a Laminate instance with specified ply angles and lamination parameters.

    The laminate is defined by weights and function values calculated based on ply angles.
    For symmetric laminates, `num_plies` represents half the total plies, with weights adjusted
    accordingly.

    Args:
        num_plies (int): Number of plies in the laminate (half the total for symmetric laminates).
        angles (Sequence[int | float]): Allowed ply angles.
        symmetric (bool, optional): Whether the laminate is symmetric. Defaults to False.
        weight_types (str, optional): Specifies which lamination parameters ('A', 'B', 'D') to calculate.
            Defaults to 'AD' if symmetric, 'ABD' otherwise.
        angle_functions (Sequence[AngleFunction], optional): Functions for calculating ply-angle values.
            If None, defaults to [cos(2x), sin(2x), cos(4x), sin(4x)].
        deg (bool, optional): Specifies if `angles` are in degrees. Defaults to False.
        funcs_round_decimals (int, optional): Rounds function values to a specified number of decimals.
            Defaults to None.

    Returns:
        Laminate: A Laminate instance configured with the specified weights and functions.
    """
    weights = generate_weights(num_plies, symmetric=symmetric, which=weight_types)
    funcs = generate_funcs(
        angles, angle_functions=angle_functions, deg=deg,
        round_decimals=funcs_round_decimals
    )
    return Laminate(weights,funcs)

