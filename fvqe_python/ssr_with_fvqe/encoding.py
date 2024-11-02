"""
encoding.py

Functions for encoding and transforming stacking sequences.

This module provides utilities to convert between three formats of stacking sequences used
in laminated composite optimization for quantum algorithms:
    1. Integer sequences for ply angles,
    2. Binary strings of '0's and '1's for representing these sequences on quantum basis states,
    3. Dense integer representations.

Functions:
    - Conversion between ply angle sequences, binary state strings, and dense integer formats.
    - Transformation of functions for encoding-specific calculations.

These tools enable seamless translation across representations, facilitating optimization
and retrieval tasks within quantum algorithms such as the F-VQE.
"""

import numpy as np
from typing import Callable, Optional
from .typing import Stack, Encoding, QubitState, QubitFunc
from .laminate import Laminate

def stack_to_state(stack: Stack, encoding: Encoding) -> QubitState:
    """Encode a stacking sequence to a qubit basis state

    Args:
        stack (Stack): The stacking sequence as a numpy array with
            integer entries 0,...,num_angles-1
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        str: The according qubit basis state

    """
    return ''.join(''.join(str(t) for t in encoding[s]) for s in stack)


def state_to_stack(state: QubitState, encoding: Encoding) -> Stack:
    """Decode a qubit basis state to a stacking sequence

    Args:
        state (str): The qubit basis state as a string of `'0'` and `'1'`
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where
            `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        Stack: The corresponding stacking sequence as a numpy array
            with the according angle indices from 0,...,`num_angles-1
            as elements

    """
    qb_per_qd = len(encoding[0])
    num_plies = len(state) // qb_per_qd
    encoding_dict = {''.join(str(t) for t in enc): s for s, enc in enumerate(encoding)}
    return np.array([
        encoding_dict[state[n * qb_per_qd:(n + 1) * qb_per_qd]] 
        for n in range(num_plies)
    ], dtype=int)

def state_to_dense(state: QubitState) -> int:
    """Convert a qubit state to bit representation

    Args:
        state (QubitState): the qubit basis state as a string of '0' and '1'

    Returns:
        int: An integer with bits corresponding to `state`
    """
    return int(state[::-1], 2)

def dense_to_state(x: int, num_qubits: int) -> QubitState:
    """Convert bit representation to qubit state

    Args:
        x (int): An integer with bits corresponding to the qubit states
        num_qubits (int): The total number of qubits

    Returns:
        QubitState: A string of '0' and '1' representing the basis state
    """
    return f'{x:0{num_qubits}b}'[::-1]

def stack_to_dense(stack: Stack, encoding: Encoding) -> int:
    """Encode stacking sequence in bit representation

    Args:
        stack (Stack): The stacking sequence as a numpy array with
            integer entries 0,...,num_angles-1
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        An integer with bits corresponding to the qubit state when encoding
        the stacking sequence 
    """
    return state_to_dense(stack_to_state(stack, encoding))

def dense_to_stack(x: int, num_qubits:int, encoding: Encoding) -> Stack:
    """Decode bit representation to stacking sequence

    Args:
        x (int): An integer with bits corresponding to the qubit states
        num_qubits (int): The total number of qubits
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        Stack: The stacking sequence as a ndarray containting the
            ply-angle indices
    """
    return state_to_stack(dense_to_state(x, num_qubits), encoding)

def convert_func_qd_to_qb(func: Callable[[Stack], float], encoding: Encoding) -> QubitFunc:
    """ Convert a function Stack -> float to a function QubitState -> float

    Args:
        func (Callable[[Stack], float]): scalar function on stacking sequences
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        QubitFunc: The corresponding function, taking the encoded
            qubit states in from of strings of '0' and '1' as an
            argument
    """
    return lambda x: func(state_to_stack(x, encoding))

def convert_func_to_dense(
    func: Callable[[Stack], float], num_qubits: int, encoding: Encoding
) -> Callable[[int],float]:
    """Convert a function Stack -> float to a function int -> float

    Args:
        func (Callable[[Stack], float]): scalar function on stacking sequences
        num_qubits (int): The total number of qubits
        encoding (Encoding): A list specifying the qubit encoding
            of the entries where `encoding[angle_idx] == (q1,q2,...)`
            specifies that the angle `angle_idx: int` is encoded
            with qubit states `(q1,q2,...)` where q1,q2,... in (0, 1)

    Returns:
        Callable[[int],float]: The corresponding function, taking the
            encoded dense bit representation in form of an integer
    """
    return lambda x: func(dense_to_stack(x, num_qubits, encoding))