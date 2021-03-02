# ============================================================================ #
# Graphics Libs
# ============================================================================ #
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
# if DARK_BACKGROUND_PLT: 
# plt.style.use('dark_background')
# plt.style.use('ggplot')
# pass

# Plotly - imports
# ----------------------------------------------- #
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.express as px

# ============================================================================ #
# Interactive Libs
# ============================================================================ #
import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(colorscale='plotly', world_readable=True)

# Show all code cells outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython.display import *
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# ============================================================================ #
# FastAi imports.
# ============================================================================ #
from fastcore.foundation import *
from fastcore.meta import *
from fastcore.utils import *
from fastcore.test import *
from nbdev.showdoc import *
from fastcore.dispatch import typedispatch
from functools import partial
import inspect

from fastcore.imports import in_notebook, in_colab, in_ipython

# Constraint imports.
# ----------------------------------------------- #
if in_colab():
    from google.colab import files
    pass

SHOW_VISDOM_RESULTS = False
if  (in_notebook() or in_ipython()) and SHOW_VISDOM_RESULTS:
    import visdom
    pass

if in_colab() or in_notebook() or in_colab():
    # Back end of ipywidgets.
    from IPython.display import display    
    import ipywidgets as widgets
    pass

