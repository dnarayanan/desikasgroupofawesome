#!/usr/bin/env python
""" init file for GADGET/AREPO reading modules 

Init procedure for simread module for reading GADGET and AREPO simulation output.
Imports all modules, but does no other init functions.

"""


__author__ = "Paul Torrey"
__copyright__ = "Copyright 2014, The Authors"
__credits__ = ["Paul Torrey"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Paul Torrey"
__email__ = "ptorrey@mit.harvard.edu"
__status__ = "Beta -- forever."


#import simread.readhaloHDF5
#import simread.readidsHDF5
#import simread.readsnapHDF5
#import simread.readtreeHDF5
#import simread.readsubfHDF5

import paul_illustris_tools.readhaloHDF5
import paul_illustris_tools.readidsHDF5
import paul_illustris_tools.readsnapHDF5 
import paul_illustris_tools.readtreeHDF5
import paul_illustris_tools.readsubfHDF5

#import readsnapHDF5
