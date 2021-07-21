"""
Code snippets for testing different binary ninja functionality.
"""
from FlowRyder import GraphNode, GraphStack, FlowRyderView
from binaryninja import *
import unittest
from codetiming import Timer

from icecream import install, ic
# Makes icecream available to all python modules imported from here
install()
ic.configureOutput(includeContext=True)


def test_bin_flow(binary_filename, source, sink):
    with Timer(name="Binja analysis"):
        program = binary_filename
        bv = BinaryViewType.get_view_of_file(program)
        view = FlowRyderView(bv=bv)

        gs = GraphStack(bv=bv, source_symbol_str=source, sink_symbol_str=sink)

    with Timer(name="Graph generation"):
        gs.generate(greedy = True)

    with Timer(name="Graph image generation"):
        digraph = gs.getDiGraph()
        view.draw_graph(digraph, file_type='pdf')

    ic(len(gs))
    ic(gs)


class Tests(unittest.TestCase):
    binary_filename = './tests/hello_frida-v2'
    source = '_main'
    sink = '_f'

    bv = BinaryViewType.get_view_of_file(binary_filename)

    fr = FlowRyderView(bv = bv)

    def bad_invalid_gn(self):
        '''Adding invalid graphnode should error graphstack
        '''
        gs = GraphStack()
        gn = GraphNode(func=self.sink_function)
        gs.add(GraphNode(func=self.sink_function))

        gs.add('ee')


    def bad_dup_gn(self):
        '''Adding duplicate graphnode to graphstack should error
        '''
        node1 = GraphNode(func = self.sink_function)
        gs = GraphStack()
        gn = GraphNode(func=self.sink_function)
        gs.add(GraphNode(func=self.sink_function))
        gs.add(gn)
        gs.add(gn)

        gs.add('ee')
        gs.add('ee')


    def good_graphstack(self, **kwargs):
        gs = GraphStack(bv = self.bv, source_symbol_str = kwargs.get('source'), sink_symbol_str = kwargs.get('sink'))
        gs.generate(greedy = kwargs.get('greedy'))

        ic.enable()
        ic(kwargs.get('source'), kwargs.get('sink'), gs)
        return gs


    def test_good_graphstack(self):
        self.assertEqual(len(self.good_graphstack(source = '_a0', sink = '_f', greedy = True)), 2)

    def test_good_graphstack2(self):
        self.assertEqual(len(self.good_graphstack(source = '_a1', sink = '_f', greedy = True)), 2)

    def test_good_graphstack3(self):
        self.assertEqual(len(self.good_graphstack(source = '_b1', sink = '_f', greedy = True)), 3)

    def test_good_graphstack4(self):
        # Slightly more complex graph with intermediary nodes
        self.assertEqual(len(self.good_graphstack(source = '_main', sink = '_f', greedy = True)), 7)


if __name__ == '__main__':
    # test_bin_flow(binary_filename, source, sink)

    unittest.main()
