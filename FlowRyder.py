'''This is an attempt to generate code a source and sink call flow graph.
Intended to run as a binja plugin.
Some code inspired from https://www.geeksforgeeks.org/find-paths-given-source-destination/
Testing only on c compiled binaries.

This plugin has been loosely been organized as an mvc.
'''
from binaryninja import *
import base64
import os
os.environ['PATH'] += os.pathsep + '/usr/local/bin/'
import graphviz
import subprocess
from codetiming import Timer
from collections import defaultdict


try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class GraphNode():
    '''This object is used for creating a connected graph displaying
    source to sinks.
    '''
    def __init__(self, **kwargs):
        self.func: binaryninja.function.function.Function = kwargs.get('func')

        if not isinstance(self.func, binaryninja.function.Function):
            raise ValueError('Must initialize node with binary ninja function type')

        self.state = ''
        self.is_sink = False
        self.is_source = False
        self.edges_to_node = set()
        self.debug = kwargs.get('debug')
        self.visited = False

        # Function callers
        # Returns a list of functions that code reference the node.
        self.edges_to_node: set(binaryninja.function.Function) = set(self.func.callers)

        # Function callees
        self.edges_from_node: set(binaryninja.function.Function) = set(self.func.callees)
        
        self.caller: GraphNode = kwargs.get('caller')

    def __str__(self) ->str:
        try:
            if self.func.symbol:
                return str(self.func.symbol.short_name.replace(':', '|'))
        except:
            return str(self.func)

    def address(self):
        return self.func.symbol.address

    def __repr__(self) ->str:
        # Overide how nodes are printed.
        try:
            if self.func.symbol:
                return str(self.func.symbol.short_name.replace(':', '|'))
        except:
            return str(self.func)

    def getEdgesToNode(self) ->set:
        for edge in set(self.edges_to_node):
            yield GraphNode(func = edge)

    def getEdgesFromNode(self) ->set:
        for edge in set(self.edges_from_node):
            yield GraphNode(func = edge)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, GraphNode):
            return GraphNode.getSymbol(self.func) == GraphNode.getSymbol(other.func)
        
        elif isinstance(other, str):
            return GraphNode.getSymbol(self.func) == other

        return False

    def __hash__(self):
        return hash(GraphNode.getSymbol(self.func))

    def getAllEdges(self) ->set:
        return self.getEdgesFromNode().union(self.getEdgesToNode())

    @staticmethod
    def getSymbol(bn_func):
        return bn_func.symbol.short_name.replace(':', '|')


class GraphStack(set):
    '''This is nothing like a stack but more like a piles.
    e.g. gs = GraphStack()
    gs.append(GraphNode(binja_function))
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()

        # completed paths will contain every possible path from a node, as a chain. 
        self.completed_paths = defaultdict(list)

        '''Maintain a blacklist of memory addr not to add to stack.
        A node is added to this set if it isn't the source, and doesn't
        have any callers.
        '''
        self.dead_nodes: GraphStack = set()

        # Objective is to discover a source to sink connected graph.
        self.source: GraphNode = None
        self.sink: GraphNode = None

        '''Use a set of graphnodes to prequailify before adding to final
        GraphStack.
        Must set empty to prevent recursion.
        '''
        self.temp_edge_nodes: GraphStack = set()

        # Debug flag for verbose print states.
        self.debug = kwargs.get('debug')
        # BinaryNinjaView
        self.bv = kwargs.get('bv')

        self._init(**kwargs)


    def _init(self, **kwargs) ->None:
        '''Generating a call graph won't work until we have sink and source.
        A binaryninja view (bv) is required to properly initialize.

        :param source_gn
        :param sink_gn
        :param source_symbol_str Unmangled function symbolic name
        :param sink_symbol_str
        :param bv as binary ninja view.
        '''
        source_gn: GraphNode = kwargs.get('source_gn')
        sink_gn: GraphNode = kwargs.get('sink_gn')
        source_symbol_str: str = kwargs.get('source_symbol_str')
        sink_symbol_str: str = kwargs.get('sink_symbol_str')
        bv: BinaryView = self.bv or kwargs.get('bv')

        if not bv:
            return

        # Validate callers parameters are either graphnodes or valid symbol names.
        if source_symbol_func := bv.get_functions_by_name(source_symbol_str).pop():
            source_gn = GraphNode(func = source_symbol_func)

        if sink_symbol_func := bv.get_functions_by_name(sink_symbol_str).pop():
            sink_gn = GraphNode(func = sink_symbol_func)

        if isinstance(source_gn, GraphNode) and isinstance(sink_gn, GraphNode):
            source_gn.is_source = True
            self.source = source_gn

            sink_gn.is_sink = True
            self.sink = sink_gn
            return
                    
        raise ValueError(f'GraphStack init item is invalid')


    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def printAllPathsUtil(self, src, dest, visited: dict, path: list, result: list, calling_node=None):
        # Mark the current node as visited and store in path
        visited.update({src:True})
        path.append(src)
        ic(src, dest, visited, path, result)
 
        # If current vertex is same as destination, then print
        # current path[]
        if src == dest:
            result.append(path.copy())
            ic(path, result, calling_node)

        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            idx, gn = self.getNode(src)
            
            # TODO: Find a way to prevent visting edges that have already been exhausted
            for src_edge in self.getConnectedEdgesFromNode(gn):
                ic(src_edge, dest, visited, path, result)
                
                if self.isCircular(src_edge):
                    ic(src_edge)

                if self.isCircular(gn):
                    ic(gn)

                if src_edge == src:
                    '''Possible that we have self calling code, but visited status will prevent loop.'''
                    ic(src_edge, src)

                if visited.get(src_edge) == False:
                    self.printAllPathsUtil(src_edge, dest, visited, path, result, src)

            self.completed_paths[src] = [True]
                     
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited.update({src:False})
        ic(visited, path)
  
  
    # Prints all paths from 's' to 'd'
    def getAllPaths(self, s, d):
        # Mark all the vertices as not visited
        # Must exlude sink to work.
        visited: dict = {x:False for x in self}

        ic(visited)

        # Create an array to store paths
        path = []
        result = []
 
        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path, result)

        ic(result)
        return result


    def getConnectedEdgesFromNode(self, gn: GraphNode) ->Generator:
        '''Get a nodes connected edges but only if the edge is present in the graph stack.

        Returns:
        Generator {GraphNode, ...}
        '''
        for bn_func in set(gn.edges_from_node):
            func_symbol = GraphNode.getSymbol(bn_func)
            ic(bn_func, func_symbol)

            if not self.isdisjoint([func_symbol]):
                yield GraphNode(func = bn_func)


    def getConnectedEdgesToNode(self, gn: GraphNode) ->Generator:
        '''Get a nodes connected edges but only if the edge is present in the graph stack.

        Returns:
        Generator {GraphNode, ...}
        '''
        for bn_func in set(gn.edges_to_node):
            func_symbol = GraphNode.getSymbol(bn_func)
            ic(bn_func, func_symbol)

            if not self.isdisjoint([func_symbol]):
                yield GraphNode(func = bn_func)


    def getDiGraphFromPaths(self, valid_paths):
        digraph = {}

        for idx, path in enumerate(valid_paths):

            for idx_node, node in enumerate(path):

                if idx_node + 1 >= len(path):
                    '''Last node in list is going to be sink'''
                    ic(idx_node, node)

                    if not digraph[path[idx_node - 1]].get(node):
                        '''Add node edge if doesn't exist in node's list'''
                        digraph[path[idx_node - 1]][node] = []

                    digraph[path[idx_node - 1]][node].append(1)

                elif idx_node == 0:
                    '''First node is source'''
                    if not digraph.get(node):
                        digraph[node] = {}
                        ic()
                
                else:
                    ic(idx_node, node)

                    if not digraph.get(node):
                        digraph[node] = {}

                    if not digraph[path[idx_node - 1]].get(node):
                       digraph[path[idx_node - 1]][node] = []

                    digraph[path[idx_node - 1]][node].append(1)
                    
            
        return ((x, digraph[x]) for x in digraph)
                    

    def getDiGraph(self) ->Generator:
        '''
        :returns
        Tuple (str, dict(list[int]))
            e.g.
            'main', {'printf':[1001, 1020]}
        '''
        digraph = {}

        for node in self:
            self.debug and print(f'Digraph node: {node}')

            for edge in node.getEdgesToNode():
                if not {edge}.isdisjoint(self):
                    '''GraphNode might have edges which are not of interest. Filtering them out.'''

                    if not digraph.get(str(node)):
                        # Create new key using node name
                        digraph[str(node)] = {}

                    if not digraph[str(node)].get(str(edge)):
                        # Create subkey using edge's name
                        digraph[str(node)][str(edge)] = []

                    digraph[str(node)][str(edge)].append(str(edge.address()))
                    self.debug and print(f'Digraph edge: {edge}')

            if digraph.get(str(node)):
                ic(digraph.get(str(node)))
                yield str(node), digraph.get(str(node))


    def append(self, item: GraphNode, p_dead_nodes: set = set()):
        '''Add a GraphNode to GraphStack
        '''
        dead_nodes = p_dead_nodes or self.dead_nodes

        if isinstance(item, GraphNode):
            if len(self) == 0:
                # First item in list will be sink
                # TODO: DELETE THIS BLOCK
                # item.is_sink = True
                pass

            # Prevent blacklisted nodes from being added to stack.
            if item in dead_nodes:
                return False
            else:
                super().add(item)

        elif isinstance(item, binaryninja.function.Function):
            gn = GraphNode(func = item)

            if len(self) == 0:
                # First item in list will be sink
                gn.is_sink = True
            
            # Prevent blacklisted nodes from being added to stack.
            if gn in dead_nodes:
                return False
            else:
                super().add(gn)

        elif all(type(x) is binaryninja.function.Function for x in item):
            for bn_func in item:
                gn = GraphNode(func = bn_func)

                if len(self) == 0:
                    # First item in list will be sink
                    gn.is_sink = True
                
                # Prevent blacklisted nodes from being added to stack.
                if gn in dead_nodes:
                    return False
                else:
                    super().add(gn)

        else:
            raise ValueError(f'Appending item is wrong type')


    def addNodes(self, **kwargs):
        '''Remove item from temp stack and determine if node should go to perm stack, and
        whether the node's edges are eligible to populate temp stack, to continue descent.
        '''
        # Greedy indicates that caller wishes to explore all paths from source to sink.
        greedy = kwargs.get('greedy', False)
        debug = kwargs.get('debug', self.debug)

        self.debug and ic.enable()
        ic.disable()

        # Get ref to a graphnode of stack.
        gn: GraphNode = self.temp_edge_nodes.pop()
        ic(gn)

        # Stop if node is source because we don't want edges calling source node.
        if gn == self.source:
            self.append(gn)

            if not greedy:
                # Wipe out temp stack to stop generate()
                self.temp_edge_nodes = set()

            return
            '''There may be more than one path from source to sink
            '''

        if temp_edges := gn.getEdgesToNode():
            # Pull all connected edges from current node
            ic(f'Edges for {gn}: {temp_edges}')

            # Prevent any edges that are already on stack from continuing
            # TODO: Fix BigO

            for temp_edge in temp_edges:
                if {temp_edge}.isdisjoint(self):
                    self.temp_edge_nodes.append(temp_edge, self.dead_nodes)

            # Move temp node to perm stack
            self.append(gn)
        else:
            # blacklist to prevent being added to gs.
            ic(f'Dead end: {gn}')
            self.dead_nodes.add(gn)
    

    def getNode(self, node_value):
        '''Returns:
        index,   type integer
        node   type string
        '''
        for idx, node in enumerate(self):
            if node == node_value:
                return idx, node  


    def generate(self, **kwargs):
        '''Expect to call this after source and sink defined. At least one graphnode in stack.
        :param greedy=bool. Indicates caller wants maximum paths if true. Slow.
        '''
        greedy = kwargs.get('greedy', True)
        self.debug and ic.enable()
        ic.disable()

        if not (self.source or self.sink):
            # Source and sink should at least be defined.
            raise ValueError(f'Please init() stack with source and sink.')

        # Initialize our temp stack
        self.temp_edge_nodes = GraphStack()

        # sink is the root node
        self.temp_edge_nodes.append(self.sink)

        with Timer(name="context manager"):
            self.addNodes(**kwargs)

            while (len(self) > 0 and len(self.temp_edge_nodes) > 0):
                # Let's collect some edges for sink
                self.addNodes(**kwargs)
                ic(f'self {self} size: {len(self)}')
                ic(f'temp {self.temp_edge_nodes} size: {len(self.temp_edge_nodes)}')

        ''' At this point, the working temp stack is empty and all nodes with path to sink
        are stored in gs. gs has more nodes than there probably should be.
        Will now use gs nodes as starting point to copy over to temp stack. This will be
        our starting point. Goal is to figure out which nodes are not needed e.g. functions
        that get called by never call other nodes connected to graph.
        '''
        ic(f'Clean graph')
        self.temp_edge_nodes_prune = list(self)
        
        # Turn gs into list because we need control over order of items.

        first_node_check = self.temp_edge_nodes_prune[0]
        old_prune_list_sz = len(self.temp_edge_nodes_prune)

        ic.enable()
        ic(len(self))
        ic.disable()
        # ic(f'temp_edge_nodes_ascend {self.temp_edge_nodes_prune}')        

        with Timer(name="context manager"):
            while (len(self.temp_edge_nodes_prune) > 0):

                self.prune()

                if len(self.temp_edge_nodes_prune) < old_prune_list_sz:
                    '''List has pruned, so reset size.'''
                    old_prune_list_sz = len(self.temp_edge_nodes_prune)

                    if self.temp_edge_nodes_prune:
                            first_node_check = self.temp_edge_nodes_prune[0]

                else:
                    '''At this point every ele should have been checked.'''

                    if self.temp_edge_nodes_prune[0] == first_node_check:
                        '''The first element is the same and list hasn't shrunk. We can't reduce anymore.'''
                        break


    def add(self, element):
        '''Overide parent method.
        '''
        return self.append(element)
        

    def prune(self) -> None:
        '''Remove non-valid node paths to sink.
        Go through gs and test nodes for dead end.
        '''
        ic.disable()
        gn: GraphNode = self.temp_edge_nodes_prune.pop()

        ic(f'Popped gn: {gn}')

        if gn in (self.source, self.sink):
            # Stop if node is source or sink
            ic(f'Prune to sink {gn.is_sink} or source {gn.is_source}.')
            return

        if edges := gn.getEdgesToNode():
            # Test if node has edges directed to it and those edges are in gs.
            ic(f'Edges to {gn}: {edges}')

            if self.isCircular(gn):
                ic(f'Circular reference.')
                self.dead_nodes.add(gn)
                self.remove(gn)
                return
            
            for edge in edges:

                if not {edge}.isdisjoint(self) and {edge}.isdisjoint(self.temp_edge_nodes_prune):
                    # Test if node's edge(s) are in gs but not in temp stack. i.e. One of it's edges has passed 
                    # connected test.
                    ic(f"Node's edge(s) are in gs but not in temp stack")
                    return

                elif edge == self.source:
                    '''Node is being called by source node. No need to test node again.'''
                    ic("Gn's Edge is source")
                    return

                elif edge == gn:
                    ic(f'Self reference.')

                elif len(self.temp_edge_nodes_prune) == 1 and not {edge}.isdisjoint(self):
                    # TODO: Test for last node in temp stack
                    ic(f'Last node in temp stack with connected edges to gs.')

                elif not {edge}.isdisjoint(self.dead_nodes):
                    ic(f'Edge {edge} is in deadend.')

                elif not {edge}.isdisjoint(self.temp_edge_nodes_prune):
                    # An edge could be connected.
                    # Not sure yet, so put gn back into the queue to reassess later.
                    # TODO: Look for a better test.
                    self.temp_edge_nodes_prune.insert(0, gn)
                    return

            # None of the edges connected to node are sink/source or part of temp stack.
            # blacklist to prevent being added to gs.
            ic(f"Edge check resulted in removal of gn")
            self.dead_nodes.add(gn)
            self.remove(gn)

        else:
            # blacklist to prevent being added to gs.
            ic(f"No edges so must be deadends")
            self.dead_nodes.add(gn)
            self.remove(gn)


    def isCircular(self, node: GraphNode) ->bool:
        # TODO: Add more complex test cases.
        if edges_to_node := list(node.getEdgesToNode()):
            if len(edges_to_node) == 1:
                edge_to_node = edges_to_node.pop()
                if  edge_to_edge := list(edge_to_node.getEdgesToNode()):
                    if len(edge_to_edge) == 1:
                        if node == edge_to_edge.pop():
                            # Test if single edge to node is the node itself.
                            return True


class FlowRyder(BackgroundTaskThread):
    def __init__(self, bv, function, *args, **kwargs):
        BackgroundTaskThread.__init__(self, '', True)
        self.progress = "FlowRyder Running..."

        self.bv = bv
        self.function = function
        # demangle type can be cppfilt, bn, None
        self.demangle = kwargs.get('demangle')
        '''Caller can specify which method to invoke.'''
        self.method = kwargs.get('method')
        self.debug = kwargs.get('debug')


    def run(self):
        if self.function:
            self.view_flowgraph_to_function()


    def view_flowgraph_to_function(self):
        bv = self.bv
        view = FlowRyderView(bv=self.bv)
        flowgraph = {}
        
        if self.function:
            sink_symbol_func = self.function
        else:
            raise ValueError(f'Not a valid function for a sink.')

        while True:
            try:
                source = get_text_line_input("Please specify source function", "Source")
                self.debug and print(f'{source}')
                gs = GraphStack(bv=bv, sink_symbol_str=sink_symbol_func.symbol.short_name, source_symbol_str=source)

            except ValueError as err:
                print(f'Invalid source symbol name: {err}')
                continue

            break

        gs.generate(greedy=True)
        ic(len(gs))
        ic(gs)

        view_graph = gs.getDiGraph()

        view.draw_graph(view_graph)


class FlowRyderView():
    def __init__(self, **kwargs):
        self.GRAPHVIZ_OUTPUT_PATH = '/tmp/'
        self.debug = False
        self.bv = kwargs.get('bv')


    def get_styles(self, label):
        styles = {
            'graph': {
                'label': label,
                'fontsize': '16',
                'fontcolor': 'white',
                #'bgcolor': '#333333',
                'bgcolor': '#101010',
                #'rankdir': 'LR',
            },
            'nodes': {
                'fontname': 'Helvetica',
                'shape': 'box',
                'fontcolor': 'white',
                'color': 'white',
                'style': 'filled',
                'fillcolor': '#006699',
            },
            'edges': {
                #'style': 'dashed',
                'color': 'white',
                'arrowhead': 'open',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'white',
            }
        }
        return styles


    def apply_styles(self, graph, styles):
        graph.graph_attr.update(('graph' in styles and styles['graph']) or {})
        graph.node_attr.update(('nodes' in styles and styles['nodes']) or {})
        graph.edge_attr.update(('edges' in styles and styles['edges']) or {})
        return graph


    def __fix_aspect_ratio(self, filename, file_type):
        '''Hack to unflatten aspect ratio of graphviz graph images.
        '''
        unflatten = subprocess.Popen(['unflatten', '-f', '-l4', '-c6', filename],
                                stdout=subprocess.PIPE,
                                )

        dot = subprocess.Popen(['dot'],
                                stdin=unflatten.stdout,
                                stdout=subprocess.PIPE,
                                )

        gvpack = subprocess.Popen(['gvpack', '-array_t6'],
                                stdin=dot.stdout,
                                stdout=subprocess.PIPE,
                                )

        neato = subprocess.Popen(['neato', '-s', '-n2', '-T' + file_type, '-o' + filename + '.' + file_type],
                                stdin=gvpack.stdout,
                                stdout=subprocess.PIPE,
                                )

        end_of_pipe = neato.stdout

        for line in end_of_pipe:
            print('\t', line.strip())
            

    def __draw_graph(self, flowgraph, function=None, filename=None, forwards=False, **kwargs):
        '''
        Params:
        flowgraph   Generated dictionary from getDiGraph()

        Returns:
            Graphviz graph object.

            filename.
        '''
        file_type = kwargs.get('file_type','png')
        '''Iterating over every xref can clutter a graph.
         Better to display one arrow and label with count.
        '''
        xref_style = 'count'
        gv_filename_path = None

        if filename == None:
            # Caller hasn't specified a filename, so generate one
            if function and hasattr(function, 'symbol'):
                # Are we displaying a function level graph, from gui?
                func_symbol = self.__get_demangled(function.symbol.name)
                filename = os.path.basename(self.bv.file.filename)
                filename = f"{filename}-{func_symbol}"

            elif self.bv.file.filename:
                # Append function symbol to filename
                filename = os.path.basename(self.bv.file.filename)

            else:
                # Append function symbol to filename
                filename = os.path.basename('FLOWRYDER-OUTPUT')

        g = graphviz.Digraph(format=file_type,
            directory=self.GRAPHVIZ_OUTPUT_PATH,
            filename=filename,
            graph_attr={'nodesep': '2.0'},
        )

        for node, value in flowgraph:
            g.node(str(node), color='blue')
            dst = node

            for src in value.keys():
                count_label = len(value.get(src, []))
                # Used to display count of xrefs between nodes

                if xref_style == 'count':
                    if forwards:
                        g.edge(str(dst), str(src), label=str(count_label))
                    else:
                        g.edge(str(src), str(dst), label=str(count_label))
                else:
                    for xref_addr in flowgraph[node][src]:
                        self.debug and print('xref_addr: {}'.format(xref_addr))

                        if forwards:
                            g.edge(dst, str(src), label=hex(xref_addr).replace("L", ""))
                        else:
                            g.edge(src, src(dst), label=hex(xref_addr).replace("L", ""))

        ic(g)

        styles = self.get_styles('Flowgraph {}'.format(filename))
        g = self.apply_styles(g, styles)

        try:
            if g.render():
                gv_filename_path = os.path.join(self.GRAPHVIZ_OUTPUT_PATH, filename)
                # Improve aspect ratio of graph.
                self.__fix_aspect_ratio(gv_filename_path, file_type)
                # Graphviz automatically appends file type extension
                return g, f'{gv_filename_path}.{file_type}'

        except IsADirectoryError as err:
            print(f'Error saving file {gv_filename_path}.{file_type}')
            

    def draw_graph(self, flowgraph, function=None, forwards=False, **kwargs) ->None:
        '''Takes a flowgraph and displays the graphic.

		:param Dictionary flowgraph: e.g. {'main':{'printf':[1001, 1020]} ...}
        :param {file_type: str}

		:rtype: None
        '''
        filename = None
        graphviz_digraph = None

        if kwargs.get('file_type') == 'pdf':
            (graphviz_digraph, filename) = self.__draw_graph(flowgraph, function=function, forwards=forwards, **kwargs)
            show_message_box('Flow saved to', f'{filename}')

        elif kwargs.get('file_type') == 'png':
            (graphviz_digraph, filename) = self.__draw_graph(flowgraph, function=function, forwards=forwards, **kwargs)
    
            if filename:
                pngdata = base64.b64encode(open(filename,'rb').read())

                output = f"""
                <html>
                <title>Flowgraph</title>
                <body>
                <div align='center'>
                    <h1>Flowgraph</h1>
                </div>
                <div align='center'>
                    <img src='data:image/png;base64,{pngdata.decode('ascii')}' alt='flowgraph'>
                </div>
                
                </body>
                </html>
                """

                ic(output)

                self.bv.show_html_report("Flowgraph", output)
            else:
                print(f'draw_graph() failed.')
        else:
            kwargs['file_type'] = self.get_file_type_choice()
            self.draw_graph(flowgraph, **kwargs)


    def get_file_type_choice(self) -> str:
        display_choice = get_choice_input("Select graph view file type", "choices", ["png", "pdf"])
        file_type = None

        if display_choice == 0:
            file_type = 'png'
        elif display_choice == 1:
            file_type = 'pdf'

        return file_type
