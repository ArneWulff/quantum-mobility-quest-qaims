"""
parameterized_state.py

Implements a Numpy-based state vector simulator for parameterized quantum circuits (VQCs).

This module defines the `ParameterizedState` class and associated functions to construct and evaluate 
quantum circuits containing only parameterized RY and CNOT gates. It provides efficient statevector 
simulation and sampling from quantum states for various quantum circuit types.

Key components:
- **ParameterizedState**: Core class that builds and evaluates quantum circuits layer-by-layer.
- **Circuit Constructors**: Functions to generate layers for specific circuit types, like `build_hwe_circuit_layers` 
  and `build_mera_circuit_layers`, which structure the circuit layers for HWE, MERA, and repeated MERA circuits.

This module is designed for simulations of quantum circuits with fixed gate types, using 
Numpy's tensor operations for efficient computation.
"""

from typing import TypeAlias, Callable, Sequence, Optional
import numpy as np
from numpy.typing import NDArray
from .typing import CircuitParams
from .counts import Counts

ParameterizedStateLayers: TypeAlias = list[tuple[str,list[tuple[int,int]]]]

class ParameterizedState:
    """Implements a quantum circuit containing layers of CNOT gates
    and parameterized RY gates.

    This class allows adding layers of RY and CNOT gates, then evaluating the
    statevector based on given parameter values and sampling from the resulting
    quantum state. Internally, the state is handled as a `num_qubits`-dimensional
    tensor (numpy array).

    Args:
        num_qubits (int): Number of qubits in the quantum circuit.
        num_params (int): Number of parameters used for the RY gate rotations.

    Attributes:
        num_qubits (int): Number of qubits in the circuit.
        num_params (int): Number of parameters for the RY gate rotations.
        layers (ParameterizedStateLayers): A sequence of layers in the form ('ry', list of gates) or ('cnot', list of gates).
    """    
    def __init__(self, num_qubits: int, num_params: int):
        self.num_qubits = num_qubits
        self.num_params = num_params
        self.layers: ParameterizedStateLayers = []  # Stores the sequence of layers (either Ry or CNOT)
        self.basis_states = np.arange(2**self.num_qubits)
        self.basis_states.flags.writeable = False

    def add_ry_layer(self, ry_gates: Sequence[tuple[int, int]]):
        """Adds an RY layer to the circuit.

        Args:
            ry_gates (Sequence[tuple[int, int]]): A sequence of (qubit_index, param_index) tuples representing
                RY gates for specific qubits and parameters.
        """
        self.layers.append(('ry', list(ry_gates)))

    def add_cnot_layer(self, cnot_gates: Sequence[tuple[int, int]]):
        """Adds a CNOT layer to the circuit.

        Args:
            cnot_gates (Sequence[tuple[int, int]]): A sequence of (control_qubit, target_qubit) tuples representing
                CNOT gates for specific qubits.
        """
        self.layers.append(('cnot', list(cnot_gates)))

    def get_last_parameter_indices(self) -> NDArray[int]:
        """Returns the indices of the last parameterized gates on each qubit.

        Returns:
            NDArray[int]: Array of indices of the last parameterized RY gates for each qubit.
        """
        set_qubits = set(range(self.num_qubits))
        parameters = set()
        for gate_type,gates in self.layers[::-1]:
            if gate_type != 'ry':
                continue
            for q,p in gates:
                if q not in set_qubits:
                    continue
                parameters.add(p)
                set_qubits.remove(q)
                if len(set_qubits) == 0:
                    break
            if len(set_qubits) == 0:
                break

        parameters = list(parameters)
        parameters.sort()
        return np.array(parameters)
    
    def apply_ry_layer(self, state: NDArray[float], ry_gates: Sequence[tuple[int, int]], cos_sin_vals: NDArray[float]):
        """Applies an RY gate layer by directly updating the statevector.

        Args:
            state (NDArray[float]): The current statevector, represented as an array.
            ry_gates (Sequence[tuple[int, int]]): Sequence of (qubit_index, param_index) tuples representing the RY gates.
            cos_sin_vals (NDArray[float]): Precomputed cosines and sines for the parameter values.
        """
        for qubit_idx, param_idx in ry_gates:
            # Slice the state along the qubit axis
            slices_0 = [slice(None)] * self.num_qubits
            slices_1 = [slice(None)] * self.num_qubits
            slices_0[qubit_idx] = 0  # Select where qubit is 0
            slices_1[qubit_idx] = 1  # Select where qubit is 1
            
            # Apply the Ry gate
            state_0 = state[tuple(slices_0)]
            state_1 = state[tuple(slices_1)]
    
            state[tuple(slices_0)], state[tuple(slices_1)] = (
                cos_sin_vals[param_idx, 0] * state_0 - cos_sin_vals[param_idx, 1] * state_1,
                cos_sin_vals[param_idx, 1] * state_0 + cos_sin_vals[param_idx, 0] * state_1
            )
    
    
    def apply_cnot_layer(self, state: NDArray[float], cnot_gates: Sequence[tuple[int, int]]):
        """Applies a CNOT layer by modifying the statevector in place.

        Args:
            state (NDArray[float]): The current statevector, represented as an array.
            cnot_gates (Sequence[tuple[int, int]]): 
                Sequence of (control_idx, target_idx) tuples representing the CNOT gates.
        """
        for control_idx, target_idx in cnot_gates:
            # Slicing for the control qubit being 1
            # slices_control_0 = [slice(None)] * self.num_qubits
            slices_control_1 = [slice(None)] * self.num_qubits
            
            # slices_control_0[control_idx] = 0  # Control qubit is 0
            slices_control_1[control_idx] = 1  # Control qubit is 1
    
            # Extract the indices for target qubit 0 and 1, within the control qubit = 1
            slices_target_0 = slices_control_1.copy()
            slices_target_1 = slices_control_1.copy()
            
            slices_target_0[target_idx] = 0  # Target qubit is 0
            slices_target_1[target_idx] = 1  # Target qubit is 1
    
            # Perform the swap directly on the state array in place
            (
                state[tuple(slices_target_0)],
                state[tuple(slices_target_1)]
            ) = (
                state[tuple(slices_target_1)].copy(), 
                state[tuple(slices_target_0)].copy()
            )

    def evaluate(self, 
            param_values: CircuitParams, previous_vals: Optional[CircuitParams] = None, 
            previous_cos_sin_vals: Optional[NDArray[float]] = None,
            return_cos_sin_vals: bool = False
        ) -> NDArray[float] | tuple[NDArray[float], NDArray[float]]:
        """Evaluates the statevector starting from |00...0> and sequentially applying each layer.

        Args:
            param_values (NDArray): Array of parameter values for the RY gates.
            previous_vals (Optional[NDArray], optional): 
                Previously used parameter values, for incremental updates. Defaults to None.
            previous_cos_sin_vals (Optional[NDArray], optional): 
                Precomputed cosines and sines for previous parameters, if available. Defaults to None.
            return_cos_sin_vals (bool, optional): If True, also returns the cosine and sine values. 
                efaults to False.

        Returns:
            NDArray: Final statevector after applying all layers. If `return_cos_sin_vals` is True, 
            returns a tuple (state, cos_sin_vals).
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
        state[(0,) * self.num_qubits] = 1.0  # Start from |00...0>

        for layer_type, gates in self.layers:
            if layer_type == 'ry':
                self.apply_ry_layer(state, gates, cos_sin_vals)
            elif layer_type == 'cnot':
                self.apply_cnot_layer(state, gates)

        return (state, cos_sin_vals) if return_cos_sin_vals else state

    def sample(self, param_values: CircuitParams, energy_fn: Callable[[int], float], shots: int = 1000) -> dict:
        """Samples from the statevector based on the resulting probability distribution.

        Args:
            param_values (CircuitParmas): Numpy array of parameter values for the RY gates.
            energy_fn (Callable[[int], float]): Function to calculate the energy of each sampled state.
            shots (int, optional): Number of samples to draw. Defaults to 1000.

        Returns:
            Counts: An instance of the Counts class containing sampled states, counts, and energies.
        """
        
        # First, evaluate the statevector
        statevector = self.evaluate(param_values).flatten()
        
        # Calculate the square of the statevector to get the probabilities
        probabilities = statevector ** 2
        # print("probs_sum:",probabilities.sum())
        
        # Sample from self.basis_states using the probabilities
        sampled_states = np.random.choice(self.basis_states, size=shots, p=probabilities)
        
        # Use np.unique to get unique sampled states and their counts
        unique_states, counts = np.unique(sampled_states, return_counts=True)
        
        return Counts(unique_states, counts, energy_fn)


def build_hwe_circuit_layers(num_qubits: int, num_reps: int) -> ParameterizedStateLayers:
    """Builds layers for a Hardware Efficient Variational Quantum Circuit (HWE).

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_reps (int): Number of repetitions of the HWE layers.

    Returns:
        ParameterizedStateLayers: List representing the layers of the circuit, including RY and CNOT layers.
    """
    num_pars_rep = 2 * num_qubits - 2
    num_parameters = num_qubits + num_reps * num_pars_rep

    layers = [('ry',[(n,n) for n in range(num_qubits)])]
    par_counter = num_qubits
    for r in range(num_reps):
        for l in range(2):
            layers.append((
                'cnot',
                [(n,n+1) for n in range(l,num_qubits-1,2)]
            ))
            layers.append((
                'ry',   # take qubit from last layer[1], last gate[1] + 1
                [(n,par_counter+n-l) for n in range(l, layers[-1][1][-1][1] + 1)]
            ))
            par_counter += len(layers[-1][1])
    return layers

def parameterized_hwe_state(num_qubits: int, num_reps: int) -> ParameterizedState:
    """Builds a parameterized state for an HWE variational quantum circuit.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_reps (int): Number of repetitions of the HWE layers.

    Returns:
        ParameterizedState: A ParameterizedState instance representing the HWE circuit.
    """
    layers = build_hwe_circuit_layers(num_qubits, num_reps)
    num_parameters = layers[-1][1][-1][1]+1
    parameterized_state = ParameterizedState(num_qubits, num_parameters)
    parameterized_state.layers.extend(layers)
    return parameterized_state


def build_mera_circuit_layers(num_qubits: int) -> ParameterizedStateLayers:
    """Builds layers for a Multi-scale Entanglement Renormalization Ansatz (MERA) inspired circuit.

    Args:
        num_qubits (int): Number of qubits in the circuit.

    Returns:
        ParameterizedStateLayers: List representing the layers of the MERA circuit, including RY and CNOT layers.
    """
    layers = [('ry',[(n,n) for n in range(num_qubits)])]
    par_counter = num_qubits
    next_power_of_2 = int(xp().ceil(xp().log(num_qubits)))
    for r in range(1,next_power_of_2+1):
        # isometries
        cnot_dist = 2**(next_power_of_2 - r)
        layers.append(('cnot', [
            (n, n+cnot_dist) for n in range(0,num_qubits-cnot_dist,2*cnot_dist)
        ]))
        ry_layer = []
        for cx in layers[-1][1]:
            ry_layer.extend([
                (q, par_counter + i) for i,q in enumerate(cx)
            ])
            par_counter += 2
        layers.append(('ry', ry_layer))

        # disentanglers
        if r <= 1:
            continue
        
        layers.append(('cnot', [
            (n, n+cnot_dist) for n in range(cnot_dist, num_qubits-cnot_dist, 2*cnot_dist)  
        ]))
        ry_layer = []
        for cx in layers[-1][1]:
            ry_layer.extend([
                (q, par_counter + i) for i,q in enumerate(cx)
            ])
            par_counter += 2
        layers.append(('ry', ry_layer))

    return layers

def build_mera_rep_circuit_layers(num_qubits: int,num_reps: int = 1) -> ParameterizedStateLayers:
    """Builds layers for a MERA-inspired circuit with repeated layers.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_reps (int, optional): Number of repetitions of the MERA layers. Defaults to 1.

    Returns:
        ParameterizedStateLayers: List of tuples representing the layers of the repeated MERA circuit, including RY and CNOT layers.
    """
    layers = [('ry',[(n,n) for n in range(num_qubits)])]
    par_counter = num_qubits
    next_power_of_2 = int(xp().ceil(xp().log(num_qubits)))
    for rep in range(num_reps):
        for r in range(1,next_power_of_2+1):
            # isometries
            cnot_dist = 2**(next_power_of_2 - r)
            layers.append(('cnot', [
                (n, n+cnot_dist) for n in range(0,num_qubits-cnot_dist,2*cnot_dist)
            ]))
            ry_layer = []
            for cx in layers[-1][1]:
                ry_layer.extend([
                    (q, par_counter + i) for i,q in enumerate(cx)
                ])
                par_counter += 2
            layers.append(('ry', ry_layer))
    
            # disentanglers
            if r <= 1:
                continue
            
            layers.append(('cnot', [
                (n, n+cnot_dist) for n in range(cnot_dist, num_qubits-cnot_dist, 2*cnot_dist)  
            ]))
            ry_layer = []
            for cx in layers[-1][1]:
                ry_layer.extend([
                    (q, par_counter + i) for i,q in enumerate(cx)
                ])
                par_counter += 2
            layers.append(('ry', ry_layer))

    return layers

def parameterized_mera_state(num_qubits: int) -> ParameterizedState:
    """Builds a parameterized state for a MERA-inspired variational quantum circuit.

    Args:
        num_qubits (int): Number of qubits in the circuit.

    Returns:
        ParameterizedState: A ParameterizedState instance representing the MERA circuit.
    """
    layers = build_mera_circuit_layers(num_qubits)
    num_parameters = layers[-1][1][-1][1]+1
    parameterized_state = ParameterizedState(num_qubits, num_parameters)
    parameterized_state.layers.extend(layers)
    return parameterized_state

def parameterized_mera_rep_state(num_qubits: int, num_reps: int = 1) -> ParameterizedState:
    """Builds a parameterized state for a MERA-inspired circuit with repeated layers.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        num_reps (int, optional): Number of repetitions of the MERA layers. Defaults to 1.

    Returns:
        ParameterizedState: A ParameterizedState instance representing the repeated MERA circuit.
    """
    layers = build_mera_rep_circuit_layers(num_qubits, num_reps)
    num_parameters = layers[-1][1][-1][1]+1
    parameterized_state = ParameterizedState(num_qubits, num_parameters)
    parameterized_state.layers.extend(layers)
    return parameterized_state


