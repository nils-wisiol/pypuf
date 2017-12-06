from pypuf.learner.pac.dfa import DFAAngluin
from pypuf.simulation.dfa_based.dfa import DFA
from pypuf.tools import all_inputs
from numpy import array, count_nonzero


def dist(instance, model, n):
    inputs = array(list(all_inputs(n)))
    return (2 ** n - count_nonzero(array(instance.eval(inputs)) == array([x[1] for x in model.eval(inputs)]))) / 2 ** n


start_state = state_0_mod4 = DFA.State()
state_1_mod4 = DFA.State()
state_2_mod4 = DFA.State()
state_3_mod4 = DFA.State(-1)

state_0_mod4.on_true = state_1_mod4
state_1_mod4.on_true = state_2_mod4
state_2_mod4.on_true = state_3_mod4
state_3_mod4.on_true = state_0_mod4

state_0_mod4.on_false = state_0_mod4
state_1_mod4.on_false = state_1_mod4
state_2_mod4.on_false = state_2_mod4
state_3_mod4.on_false = state_3_mod4

instance = DFA(state_0_mod4)
model = DFAAngluin(instance).learn()

for s in all_inputs(4):
    #if model.eval1(s)[1] == -1 and sum([0 if x == -1 else 1 for x in s]) % 4 == 3:
    if instance.eval([s])[0] != model.eval1(s)[1]:
        print("on input %s: got %s but expected %s" % (
            s,
            "REJECT" if model.eval1(s)[1] == 1 else "ACCEPT",
            "REJECT" if instance.eval([s])[0] == 1 else "ACCEPT",
        ))

print('dist: %.4f' % dist(instance, model, 6))
