"""
parameterized_state_perm.py

Implements a Numpy-based state vector simulator for a permutation-based
variational quantum circuit (VQC) tailored to laminate ply angle encoding.

This module defines the `ParameterizedStatePerm` class for building quantum circuits that perform 
permutations on qudit states, representing ply angles in a laminate stacking sequence. The circuit 
is constructed using layers of partial swap gates that act between quantum digits encoding ply angles, 
allowing flexible and efficient permutations within the quantum state.

Key components:
- **ParameterizedStatePerm**: Core class that builds and evaluates permutation-based quantum circuits.
- **Circuit Layer Builders**: Functions to generate layers for specific permutation circuits, with each 
  layer consisting of partial swaps between ply-angle qudits.

This module is optimized for use cases involving parameterized partial swap gates with efficient Numpy
tensor operations for state manipulation.
"""

import numpy as np
from typing import Callable, Sequence, Optional, TypeAlias
from numpy.typing import NDArray
from .typing import CircuitParams
from .counts import Counts

ParameterizedStatePermLayers: TypeAlias = list[list[tuple[tuple[int,...], int]]]

class ParameterizedStatePerm:
    """Implements a quantum circuit for ply permutations, containing layers of 
    partial swap gates.

    **Attention!** This implementation is hardcoded for the encoding of the 4 ply-angle 
    indices as `[(0,0), (0,1), (1,1), (1,0)]`

    This class allows adding layers of partial swap gates between the ply qudit states, 
    then evaluating the statevector based on given parameter values and sampling from the 
    resulting quantum state. Internally, the state is handled as a `num_qubits`-dimensional
    tensor (numpy array).

    Args:
        initial_state (str): Initial qubit basis state as a string of '0' and '1'
        num_params (int, optional): Number of parameters in the circuit. Default is 0.

    Attributes:
        num_qubits (int): Number of qubits in the circuit.
        num_params (int): Number of parameters for the RY gate rotations.
        initial_state (str): The initial state as a string of '0' and '1'
        layers (ParameterizedStateLayers): A sequence of layers in the form ('ry', list of gates) or ('cnot', list of gates).
    """  
    def __init__(self, initial_state: str, num_params: int = 0):
        self.num_qubits = len(initial_state)
        self.num_params = num_params
        self.initial_state = initial_state
        self.layers: ParameterizedStatePermLayers = []
        self.basis_states = np.arange(2**self.num_qubits)
        self.basis_states.flags.writeable = False

    def add_layer(self, gates: Sequence[tuple[int,...]]):
        """Adds a new layer of gates to the circuit.

        Args:
            gates (Sequence[tuple[int, ...]]): Sequence of tuples, where each tuple represents a qudit gate
                by specifying qubit indices it acts on.
        """
        self.layers.append(gates)
        self.num_params += len(gates)

    def get_last_parameters(self):
        """Placeholder method for retrieving parameters of the last gate layer."""
        raise NotImplementedError("Method `get_last_parameters` is not yet implemented for `ParameterizedStatePerm`.")

    def apply_layer(self, state: NDArray[float], qudit_gates: list[tuple[int,...]], cos_sin_vals: NDArray[float]):
        """Applies a layer of partial swap gate, which acts on pairs of qudits with specified cosine and sine values.

        Args:
            state (NDArray[float]): The current state, represented as an array.
            qudit_gates (list[tuple[int, ...]]): List of gates, each defined by a tuple of qubit indices and a parameter index.
            cos_sin_vals (NDArray[float]): Array of cosine and sine values for each parameter, used in swap operations.
        """
       # List of state pairs to be swapped
        swap_pairs = [
            ((0, 0, 0, 1), (0, 1, 0, 0)),
            ((0, 0, 1, 1), (1, 1, 0, 0)),
            ((0, 0, 1, 0), (1, 0, 0, 0)),
            ((0, 1, 1, 1), (1, 1, 0, 1)),
            ((0, 1, 1, 0), (1, 0, 0, 1)),
            ((1, 1, 1, 0), (1, 0, 1, 1))
        ]
        
        for (qb11, qb12, qb21, qb22), param_idx in qudit_gates:
            # For each qudit gate, we perform swaps for all pairs in swap_pairs
            for state_a, state_b in swap_pairs:
                
                # Initialize slices for state_a and state_b (with all qubits being slice(None))
                slices_a = [slice(None)] * self.num_qubits
                slices_b = [slice(None)] * self.num_qubits
                
                # Set the specific qubits for qudit 1 and qudit 2 in slices_a and slices_b
                slices_a[qb11], slices_a[qb12], slices_a[qb21], slices_a[qb22] = state_a
                slices_b[qb11], slices_b[qb12], slices_b[qb21], slices_b[qb22] = state_b
                
                # Get the state components for state_a and state_b
                state_a_val = state[tuple(slices_a)]
                state_b_val = state[tuple(slices_b)]
                
                # Apply the swap with cosine and sine terms, adding minus sign for swaps in one direction
                state[tuple(slices_a)], state[tuple(slices_b)] = (
                    cos_sin_vals[param_idx, 0] * state_a_val - cos_sin_vals[param_idx, 1] * state_b_val,  # Swap from b to a has a minus sign on sin term
                    cos_sin_vals[param_idx, 1] * state_a_val + cos_sin_vals[param_idx, 0] * state_b_val   # Swap from a to b is regular
                )
        
    def evaluate(self, 
        param_values: CircuitParams, previous_vals: Optional[CircuitParams] = None,
        previous_cos_sin_vals: Optional[NDArray[float]] = None,
        return_cos_sin_vals: bool = False
    ) -> NDArray[float] | tuple[NDArray[float], NDArray[float]]:
        """Evaluates the state starting from the initial state and applying each layer.

        Args:
            param_values (CircuitParams): Array of parameter values for each gate.
            previous_vals (Optional[CircuitParams], optional): 
                Previously used parameter values, for incremental updates. Defaults to None.
            previous_cos_sin_vals (Optional[NDArray[float]], optional):
                Precomputed cosine and sine values, if available. Defaults to None.
            return_cos_sin_vals (bool, optional): If True, also returns the cosine and sine values.
                Defaults to False.

        Returns:
            NDArray[float] | tuple[NDArray[float], NDArray[float]]: Final state after applying 
                all layers. If `return_cos_sin_vals` is True, returns a tuple (state, cos_sin_vals).
                The state is an NDArray of shape `(2,) * num_qubits`.
        """
        assert (previous_vals is None) == (previous_cos_sin_vals is None)
        if previous_vals is None:
            cos_sin_vals = np.stack((
                np.cos(param_values / 2), 
                np.sin(param_values / 2)
            )).transpose()
        else:
            cos_sin_vals = previous_cos_sin_vals.copy()
            mask = param_vals != previous_vals
            cos_sin_vals[mask] = np.stack((
                np.cos(param_values[mask] / 2),
                np.sin(param_values[mask] / 2)
            )).transpose()
        
        state = np.zeros([2] * self.num_qubits, dtype=np.float64)
        state[tuple(int(bit) for bit in self.initial_state)] = 1.0  # Start from |00...0>

        for gates in self.layers:
            self.apply_layer(state, gates, cos_sin_vals)

        return (state, cos_sin_vals) if return_cos_sin_vals else state


    def sample(self, 
        param_values: CircuitParams, energy_fn: Callable[[int], float], shots: int = 1000
    ) -> dict:
        """Samples from the state based on the resulting probability distribution.

        Args:
            param_values (CircuitParams): Array of parameter values for each gate.
            energy_fn (Callable[[int], float]): Function to calculate the energy of each sampled state.
            shots (int, optional): Number of samples to draw. Defaults to 1000.

        Returns:
            Counts: An instance of the Counts class containing sampled states, counts, and energies.
        """
        
        # First, evaluate the statevector
        statevector = self.evaluate(param_values).flatten()
        
        # Calculate the square of the statevector to get the probabilities
        probabilities = statevector ** 2
        
        # Sample from self.basis_states using the probabilities
        sampled_states = np.random.choice(self.basis_states, size=shots, p=probabilities)
        
        # Use np.unique to get unique sampled states and their counts
        unique_states, counts = np.unique(sampled_states, return_counts=True)
        
        return Counts(unique_states, counts, energy_fn)

def get_qudit_gates(num_qubits: int) -> list[list[tuple[int,int]]]:
    """Generates partial swap gates on ply indices for each layer.

    Args:
        num_qubits (int): Number of qubits in the circuit.

    Returns:
        list[list[tuple[int, int]]]: A list of layers, each containing pairs of qubits representing qudit gates.
    """
    layers = []
    num_qudits = num_qubits//2
    for i in range(0,num_qudits//2):
        second_qubits = set()
        layer1 = []
        layer2 = []
        for j in range(num_qudits):
            g = (j, (j+i+1) % num_qudits)
            if j not in second_qubits:
                layer1.append(g)
                second_qubits.add(g[1])
            else:
                layer2.append(g)
        layers.append(layer1)
        if len(layer2) > 0:
            layers.append(layer2)
    # For even number of qubits, last two layers are equivalent
    return layers[:-1] if num_qubits % 2 == 0 else layers

def build_perm_circuit_layers(num_qubits: int, num_reps: int) -> tuple[ParameterizedStatePermLayers, int]:
    """Generates layers of partial swap gates for a permutation circuit.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_reps (int): Number of repetitions of the permutation layers.

    Returns:
        tuple: A tuple containing:
            - ParameterizedStatePermLayers: Layers of the permutation circuit.
            - int: The total number of parameters required by the circuit.
    """
    qd_gates = get_qudit_gates(num_qubits)
    encoding = [(0,0), (0,1), (1,0), (1,1)]
    layers = []
    par_counter = 0
    for rep in range(num_reps):
        for lr in qd_gates:
            next_layer = []
            for g in lr:
                next_layer.append(((2*g[0],2*g[0]+1, 2*g[1],2*g[1]+1), par_counter))
                par_counter += 1
            layers.append(next_layer)
    return layers, par_counter


def parameterized_perm_state(initial_state: str, num_reps: int=1):
    """Builds a parameterized state for a permutation circuit with partial qudit swaps.

    Args:
        initial_state (str): Initial qubit basis state as a string of '0' and '1'.
        num_reps (int, optional): Number of repetitions of the permutation layers. Defaults to 1.

    Returns:
        ParameterizedStatePerm: A ParameterizedStatePerm instance representing the permutation circuit.
    """
    layers,num_parameters = build_perm_circuit_layers(len(initial_state), num_reps)
    parameterized_state = ParameterizedStatePerm(initial_state, num_parameters)
    parameterized_state.layers.extend(layers)
    return parameterized_state
