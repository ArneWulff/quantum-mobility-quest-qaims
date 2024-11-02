"""
constraints.py

Defines and enforces stacking sequence constraints for laminated composites.

This module provides the `ConstraintSettings` class, which manages various stacking sequence 
constraints, including:

    - Disorientation constraints: restricts neighboring ply angles.
    - Contiguity constraints: limits consecutive plies with the same orientation.
    - Balance constraints: enforces specific angle pairs to be balanced.
    - Percent rule constraints: mandates minimum percentages for each ply angle.

The `ConstraintSettings` class calculates penalties for constraint violations within a stacking 
sequence, which is essential for composite laminate optimization.

Usage:
    Instantiate `ConstraintSettings` with constraint parameters to enable
    constraint checks and penalty calculations for stacking sequences.
"""

import numpy as np
from typing import Optional, Sequence
from numpy.typing import NDArray
from .typing import Stack
from .laminate import Laminate

class ConstraintSettings:
    """Manages constraint settings and computes constraint violations for a laminate stacking sequence.

    Args:
        laminate (Laminate): The laminate object for accessing angle and ply information.
        disorientation_matrix (Optional[NDArray[int]]): 
            Matrix specifying disallowed neighboring ply angles such that
            `disorientation_matrix[s1,s2]` is 1 if the neighboring ply-angle indices
            `s1` and `s2` violate the constraint and 0 else
        contiguity_distance (Optional[int]):
            Maximum number of contiguous plies with the same orientation.
        balanced_angles (Optional[tuple[int, int] | list[tuple[int, int]]]):
            Angle pairs that must be balanced.
        percent_rule (Optional[float | Sequence[float]]):
            Percentage of each angle required in the laminate.
        disorientation_penalty (float): Penalty multiplier for disorientation violations.
        contiguity_penalty (float): Penalty multiplier for contiguity violations.
        balanced_penalty (float): Penalty multiplier for unbalanced angle violations.
        percent_penalty (float): Penalty multiplier for percent rule violations.
    """
    def __init__(
        self,
        laminate: Laminate,
        disorientation_matrix: Optional[NDArray[int]] = None,
        contiguity_distance: Optional[int] = None,
        balanced_angles: Optional[tuple[int, int] | list[tuple[int, int]]] = None,
        percent_rule: Optional[float | Sequence[float]] = None,
        disorientation_penalty: float = 1.,
        contiguity_penalty: float = 1.,
        balanced_penalty: float = 1.,
        percent_penalty: float = 1.
    ):
        self.constraints = {
            constraint: pen for (constraint, val, pen) in [
                ('disorientation', disorientation_matrix, disorientation_penalty),
                ('contiguity', contiguity_distance, contiguity_penalty),
                ('balanced', balanced_angles, balanced_penalty),
                ('percent', percent_rule, percent_penalty)
            ] if val is not None
        }

        self.disorientation_matrix = disorientation_matrix
        self.contiguity_distance = contiguity_distance
        if balanced_angles is not None:
            if isinstance(balanced_angles[0], int):
                balanced_angles = [balanced_angles]
            else:
                balanced_angles = list(balanced_angles)
        self.balanced_angles = balanced_angles
        all_balanced_angles = set(np.array(balanced_angles).flatten())
        if isinstance(percent_rule, float):
            percent_rule = [
                percent_rule / 2 if (a in all_balanced_angles) else percent_rule
                for a in range(laminate.num_angles)
            ]
        self.percent_rule = percent_rule
        if percent_rule is not None:
            self.percent_rule_min_plies = [
                int(np.ceil(p * laminate.num_plies))
                for p in percent_rule
            ]

        self.penalties = {
            'disorientation': disorientation_penalty,
            'contiguity': contiguity_penalty,
            'balanced': balanced_penalty,
            'percent': percent_penalty,
        }

    def count_disorientation_violations(self, stack: Stack) -> int:
        """Counts disorientation constraint violations in a stack.

        Args:
            stack (Stack): The stacking sequence to evaluate.

        Returns:
            int: The total number of disorientation violations.
        """
        return int(np.sum(self.disorientation_matrix[stack[:-1], stack[1:]]))

    def count_contiguity_violations(self, stack: Stack) -> int:
        """Counts contiguity violations in a stack based on 
        consecutive plies with the same orientation.

        Args:
            stack (Stack): The stacking sequence to evaluate.

        Returns:
            int: The total number of contiguity violations.
        """
        num_violations = 0
        group_start = 0
        for n in range(1, len(stack)):
            if stack[n] != stack[group_start]:
                group_start = n
            elif n - group_start + 1 > self.contiguity_distance:
                num_violations += 1
        return num_violations

    def count_balanced_violations(self, stack: Stack) -> int:
        """Counts violations for angle pairs that must be balanced.

        Args:
            stack (Stack): The stacking sequence to evaluate.

        Returns:
            int: The total number of balance violations.
        """
        num_violations = 0
        for a1, a2 in self.balanced_angles:
            num_violations += abs(np.sum(stack == a1) - np.sum(stack == a2))
        return int(num_violations)

    def count_percent_rule_violations(self, stack: Stack) -> int:
        """Counts violations of the percent rule in the stack.

        Args:
            stack (Stack): The stacking sequence to evaluate.

        Returns:
            int: The total number of percent rule violations.
        """
        num_violations = 0
        for a, a_min in enumerate(self.percent_rule_min_plies):
            num_violations += max(0, a_min - sum(stack == a))
        return int(num_violations)

    def count_constraint_violations(self, stack: Stack) -> int:
        """Counts all types of constraint violations in the stack.

        Args:
            stack (Stack): The stacking sequence to evaluate.

        Returns:
            dict: A dictionary of constraint violations, with keys 
            as constraint names and values as counts.
        """
        constraint_violations = dict()
        for constraint in self.constraints:
            if constraint == 'disorientation':
                constraint_violations[constraint] = self.count_disorientation_violations(stack)
            elif constraint == 'contiguity':
                constraint_violations[constraint] = self.count_contiguity_violations(stack)
            elif constraint == 'balanced':
                constraint_violations[constraint] = self.count_balanced_violations(stack)
            elif constraint == 'percent':
                constraint_violations[constraint] = self.count_percent_rule_violations(stack)
        return constraint_violations

    def penalty(self, stack: Stack) -> float:
        """Computes the total penalty for a given stacking sequence 
        based on constraint violations.

        Args:
            stack (Stack): The stacking sequence to evaluate.

        Returns:
            float: The total penalty score for the stacking sequence.
        """
        violations = self.count_constraint_violations(stack)
        return float(sum([
            self.constraints[constraint] * val for constraint, val in violations.items()
        ]))