from pypuf.learner.base import Learner
from pypuf.simulation.dfa_based.dfa import DFA


class DFAAngluin(Learner):

    class ClassificationTree:

        class Node:
            def __init__(self, label, parent=None, left_child=None, right_child=None):
                self.label = label
                self.left = left_child
                self.right = right_child
                self.parent = parent

            def is_leaf(self):
                return self.left is None and self.right is None

            def is_left(self):
                if self.parent is None:
                    return False
                return self.parent.left == self

            def is_root(self):
                return self.parent is None

            def is_left_of_root(self):
                if self.is_root():
                    return False
                current = self
                while not current.parent.is_root():
                    current = current.parent
                return current.is_left()

            def access_string(self):
                assert self.is_leaf()
                return self.label

            def distinguishing_string(self):
                assert not self.is_leaf()
                return self.label

        def __init__(self, root):
            self.root = root
            self.nodes = self.find_nodes()

        def find_nodes(self, start='root'):
            if start == 'root':
                start = self.root
            if start is None:
                return []
            return self.find_nodes(start=start.left) + self.find_nodes(start=start.right)

        def replace(self, leaf_node, left_access_string, right_access_string, label):
            leaf_node.left = __class__.Node(left_access_string, parent=leaf_node)
            leaf_node.right = __class__.Node(right_access_string, parent=leaf_node)
            leaf_node.label = label
            self.nodes += [leaf_node.left, leaf_node.right]

        def leaves(self):
            return [node for node in self.nodes if node.is_leaf()]

        def print_node(self, node=None, level=0):
            if node is None:
                node = self.root
            print('\t' * level + ('L: ' if node.is_left() else 'R: ') + str(node.label))
            if node.right:
                self.print_node(node=node.right, level=level + 1)
            if node.left:
                self.print_node(node=node.left, level=level + 1)

    def __init__(self, instance):
        self.instance = instance
        self.counter_example = None
        self.hypothesis = None
        self.classification_tree = None

    def learn(self):
        # abbreviation
        instance = self.instance

        # initialization
        self.hypothesis = DFA(DFA.State(accepting=([], instance.eval1([])), on_true='self', on_false='self'))
        counter_example = self.find_counter_example()
        if counter_example is None:
            return  # We're done! :-)

        root_node = DFAAngluin.ClassificationTree.Node(label=[])
        self.classification_tree = DFAAngluin.ClassificationTree(root_node)
        if instance.eval1([]) == +1:
            # counter_example is accepting, root node is rejecting
            self.classification_tree.replace(root_node, [], counter_example, [])
        else:
            self.classification_tree.replace(root_node, counter_example, [], [])

        # main loop
        i = 0
        while i < 2:
            self.classification_tree.print_node()
            i += 1
            self.hypothesis = self.tentative_hypothesis()
            counter_example = self.find_counter_example()
            if counter_example is None:
                return
            self.update_tree(counter_example)

        print("done")
        self.hypothesis = self.tentative_hypothesis()
        return self.hypothesis

    def find_counter_example(self):
        # TODO replace this by something that is appropriate in general
        return [-1, -1, +1, -1]

    def update_tree(self, counter_example):
        print("updating tree with counter example")
        for prefix in [counter_example[:i] for i in range(len(counter_example) + 1)]:
            classification_tree_access_string = self.sift(prefix).access_string()
            hypothesis_access_string, hypothesis_acceptance = self.hypothesis.eval1(prefix)
            print("  prefix: %s hypothesis: %s, classification tree: %s" % (prefix, hypothesis_access_string, classification_tree_access_string))
            if hypothesis_access_string != classification_tree_access_string:
                # discovered a new equivalence class still unknown in the tree
                # update the tree to include the newly found class
                node_to_be_replaced = self.sift(prefix[:-1])
                self.classification_tree.replace(
                    node_to_be_replaced,
                    node_to_be_replaced.access_string(),
                    prefix[:-1],
                    [prefix[-1]] + self.find_distinguishing_string(prefix[:-1])
                )
                return
        assert False, "This should never happen."

    def find_distinguishing_string(self, prefix):
        # TODO correct implementation
        if len(prefix) == 3:
            return []
        elif len(prefix) == 1:
            return [-1]
        assert False, "Don't know how to compute distinguishing string. :-("

    def sift(self, s):
        current = self.classification_tree.root

        while not current.is_leaf():
            d = current.distinguishing_string()

            if self.instance.eval1(s + d) == -1:
                current = current.right
            else:
                current = current.left

        print("  sifting %s to %s" % (s, current.access_string()))
        return current

    def tentative_hypothesis(self):
        print("Computing hypothesis")
        states = {}
        for leaf in self.classification_tree.leaves():
            states[self.array_to_pm_string(leaf.access_string())] = DFA.State(
                accepting=
                    (
                        leaf.access_string(),
                        +1 if leaf.is_left_of_root() else -1,
                    ),
            )

        for access_string, state in states.items():
            state.on_true = states[self.array_to_pm_string(self.sift(self.pm_string_to_array(access_string) + [-1]).access_string())]
            state.on_false = states[self.array_to_pm_string(self.sift(self.pm_string_to_array(access_string) + [+1]).access_string())]
            assert state.on_false is not None
            assert state.on_true is not None

        return DFA(states[""])

    def array_to_pm_string(self, bits):
        return "".join(['+' if b == 1 else '-' for b in bits])

    def pm_string_to_array(self, string):
        bits = []
        for b in string:
            bits.append(-1 if b == '-' else +1)
        return bits
