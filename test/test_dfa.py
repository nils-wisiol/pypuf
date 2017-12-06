"""
Testing the DFA implementation of the simulation package.
"""
import unittest
import pypuf.simulation.dfa_based.dfa as dfa
from pypuf.tools import all_inputs
from numpy.testing import assert_array_equal


class TestState(unittest.TestCase):
    """
    Testing the DFA state class.
    """

    def test_init(self):
        """
        Checking if initialization works as expected
        """
        state = dfa.DFA.State(accepting=+1, on_false=dfa.DFA.State(), on_true='self')
        self.assertEqual(state.accepting, +1)
        self.assertEqual(type(state.on_false), dfa.DFA.State)
        self.assertEqual(state.on_true, state)

        state = dfa.DFA.State()
        self.assertEqual(state.accepting, +1)
        self.assertEqual(state.on_true, None)
        self.assertEqual(state.on_false, None)

        state = dfa.DFA.State(accepting=-1)
        self.assertEqual(state.accepting, -1)

        with self.assertRaises(Exception):
            dfa.DFA.State(accepting='trying to be cool')

        with self.assertRaises(Exception):
            dfa.DFA.State(on_true='asdf')

    def test_eval(self):
        """
        Checking if evaluation of state input works as expected
        """
        an_empty_state = dfa.DFA.State()
        state = dfa.DFA.State(accepting=+1, on_false=an_empty_state, on_true='self')
        self.assertEqual(state.next(+1), an_empty_state)
        self.assertEqual(state.next(-1), state)

        with self.assertRaises(Exception):
            state.next('foobar')


class TestDFA(unittest.TestCase):
    """
    Testing the DFA
    """

    def test_init(self):
        """
        Testing if DFA initializes correctly.
        """
        an_empty_state = dfa.DFA.State()
        automaton = dfa.DFA(an_empty_state)
        self.assertEqual(automaton.start_state, an_empty_state)

        with self.assertRaises(Exception):
            dfa.DFA('foobar')

    def test_eval(self):
        """
        Testing if DFA evaluates one input string correctly
        """

        # automaton accepting nothing
        empty_automaton = dfa.DFA(dfa.DFA.State())

        # automaton accepting everything
        all_automaton = dfa.DFA(dfa.DFA.State(accepting=-1, on_false='self', on_true='self'))

        # automaton accepting all strings that contain a +1
        including_1_automaton = dfa.DFA(dfa.DFA.State(
            on_false=dfa.DFA.State(accepting=-1, on_true='self', on_false='self'),  # accepting sink
            on_true='self'  # stay
        ))

        # automaton accepting all strings beginning in -1
        starting_minus1_automaton = dfa.DFA(dfa.DFA.State(
            on_true=dfa.DFA.State(accepting=-1, on_true='self', on_false='self'),  # accepting sink
            on_false=None,  # rejecting sink
        ))

        # checking if automatons behave correctly
        for input_string in all_inputs(5):
            self.assertEqual(empty_automaton.eval1(input_string), +1)
            self.assertEqual(all_automaton.eval1(input_string), -1)
            self.assertEqual(including_1_automaton.eval1(input_string), -1 if 1 in input_string else +1)
            self.assertEqual(starting_minus1_automaton.eval1(input_string), -1 if input_string[0] == -1 else +1)

        # checking if eval behaves as eval1
        automatons = [empty_automaton, all_automaton, including_1_automaton, starting_minus1_automaton]
        for automaton in automatons:
            assert_array_equal(
                [automaton.eval1(input_string) for input_string in all_inputs(4)],
                automaton.eval(all_inputs(4))
            )
