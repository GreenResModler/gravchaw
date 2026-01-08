"""
"gravchaw" is a Python framework for assimilating time-lapse gravity data into groundwater modeling. It includes modules 
for coupled modeling, an interface to the coupled model, and tools to run optimization and uncertainty analysis algorithms, 
followed by post-optimization parameter updates.

"""

from .version import __version__

from . import model_bridge
from . import execute_workflow
from . models import (gravi4gw_hybrid, model_grid, model_out, coupled_model_grid, coupled_model_out)


__all__ = ["model_bridge",
           "execute_workflow",
           "gravi4gw_hybrid",
           "model_grid", 
           "model_out",
           "coupled_model_grid",
           "coupled_model_out",
           "__version__"]
