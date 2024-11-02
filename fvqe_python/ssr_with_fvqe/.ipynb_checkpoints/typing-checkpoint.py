"""
typing.py

Definition of type aliases for type annotations

Type Aliases:
    StackElem: int
        Ply-angle indices, integers ranging from 0 to num_angles-1.
    Stack: NDArray[StackElem]
        Stacking sequence, ndarray shaped (num_plies,).
    FuncArray: NDArray[float]
        Function values f_l(s) of ply-angles, for the calculation of lamination parameters,
        ndarray shaped (num_angles, num_funcs).
    WeightsArray: NDArray[float]
        Weights alpha_n^X for the calculation of the lamination parameters,
        ndarray shaped (num_weights, num_plies).
    AngleFunction: Callable[[int | float], float]
        Function accepting an angle (in radians) and returning a float.
    Parameters: NDArray[float]
        Lamination parameters v_l^X, ndarray shaped (num_weights, num_funcs)
    Constraint: Callable[[Stack], bool]
        Constraints on stacking sequences. Returns True if the stacking sequence
        satisfies the constraint, and False otherwise.
    QubitState: str
        Basis state of the qubits, represented by a string of '0's and '1's.
    QubitFunc: Callable[[QubitState], float]
        Function on the basis states, returning a float
    CountsDict: dict[QubitState, int]
        Dictionary for collecting counts of measured basis states
    FilterFunc: Callable[[float, float], float]
        Filter function f(E,tau) for F-VQE, where E is the energy and tau
        is a tunable parameter, returns a float.
    Encoding: Sequence[tuple[int, ...]]
        Encoding of the ply-angle indices into qubit states, where
        `encoding[angle_idx]` returns a tuple of integers 0 and 1
        corresponding to the conversion of a qudit state to qubits.
        Of length num_angles, and all tuples must have the same length.
    CircuitParams: NDArray
        Parameters for a variational quantum circuit, ndarray with ndims = 1
    ParIndsFunc: Callable[[int], None | Sequence[int] | Sequence[Sequence[int]]]
        Function that takes the current iteration of an optimizer as an integer
        and returns a sequence of integer indices to define the subset of
        circuit parameters that are optimized in this iteration.
        The function may also return multiple disjoint subsets of parameter
        indices. `None` stands for including all parameters in this iteration.
"""

from typing import TypeAlias, Callable, Sequence
from numpy.typing import NDArray

StackElem: TypeAlias = int
Stack: TypeAlias = NDArray[StackElem]
FuncArray: TypeAlias = NDArray[float]
WeightsArray: TypeAlias = NDArray[float]
AngleFunction: TypeAlias = Callable[[int | float], float]
Parameters: TypeAlias = NDArray[float]
Constraint: TypeAlias = Callable[[Stack], bool]
QubitState: TypeAlias = str
QubitFunc: TypeAlias = Callable[[QubitState], float]
CountsDict: TypeAlias = dict[QubitState, int]
FilterFunc: TypeAlias = Callable[[float, float], float]
Encoding: TypeAlias = Sequence[tuple[int, ...]]
CircuitParams: TypeAlias = NDArray
ParIndsFunc: TypeAlias = Callable[[int], None | Sequence[int] | Sequence[Sequence[int]]]