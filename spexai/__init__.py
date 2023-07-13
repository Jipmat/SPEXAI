"""
SpexAI

An neural network emulator for SPEX's CIE model.
"""

__version__ = "0.0.1"
__author__ = 'Jip Matthijsse'
__credits__ = 'SRON / Universiteit van Amsterdam'


# import inference code at top-level for easier use
from spexai.inference.model import *
from spexai.inference.fit import *
from spexai.inference.write_tensors import *

