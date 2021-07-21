from binaryninja import *
from .FlowRyder import FlowRyder


def __flowryder(bv, function):
    flowryder = FlowRyder(bv, function)
    flowryder.start()

# UI menu items
PluginCommand.register_for_function(
    "FlowRyder\\Generate",
    "",
    __flowryder
)