"""
This module provides a simulation class for Boolean functions based that can be represented by DFAs, Deterministic
Finite Automatons.
"""
from pypuf.simulation.base import Simulation


class DFA(Simulation):
    """
    Class representing DFAs. Probably not the fastest implementation seen on earth. Whenever state is `None`, this will
    be considered an implicit, non-accepting sink-state.
    """

    start_state = None

    class State:
        """
        State of a DFA. Is accepting or not, and can evaluate which state comes after this on any given input.
        """

        def __init__(self, accepting=+1, on_true=None, on_false=None):
            """
            Initialize a DFA state.
            :param accepting: Use -1 for accepting states, +1 for rejecting states.
            :param on_true: The state we go to whenever -1 is input. Use None for a rejecting sink, use 'self' to stay.
            :param on_false: The state we go to whenever +1 is input. Use None for a rejecting sink, use 'self' to stay.
            """
            assert accepting in [-1, +1]
            assert on_true is None or isinstance(on_true, DFA.State) or on_true == 'self'
            assert on_false is None or isinstance(on_false, DFA.State) or on_false == 'self'
            self.accepting = accepting
            self.input_true = self if on_true == 'self' else on_true
            self.input_false = self if on_false == 'self' else on_false

        def next(self, input_bit):
            """
            :param input_bit: Input to a DFA in this state.
            :return: The new state of the DFA.
            """
            if input_bit == -1:
                return self.input_true
            elif input_bit == +1:
                return self.input_false
            else:
                assert input_bit in [-1, +1], "input bits to the DFA must be represented as +-1 bits, -1 being true"

    def __init__(self, start_state):
        """
        Initialize a DFA object with given start state.
        :param start_state: Start state of the automaton.
        """
        assert isinstance(start_state, DFA.State)
        self.start_state = start_state

    def eval(self, inputs):
        """
        Evaluate the acceptance of the DFA on a list of inputs.
        :param inputs: List of inputs given as array of arrays of bits in +-1 representation, -1 being true.
        :return: List of acceptances/rejections given as array of bits in +-1 representation, -1 being true.
        """
        return [self.eval1(input_string) for input_string in inputs]

    def eval1(self, input_string):
        """
        Evaluate the acceptance of the DFA on **one** input string.
        :param input_string: input string given as an array of bits in +-1 representation, -1 being true.
        :return: acceptance or rejection in +-1 representation, -1 meaning acceptance
        """
        current_state = self.start_state
        for bit in input_string:
            current_state = current_state.next(bit) if current_state is not None else None
        return current_state.accepting if current_state is not None else +1
