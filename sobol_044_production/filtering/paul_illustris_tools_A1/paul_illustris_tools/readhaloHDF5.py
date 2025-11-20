#!/usr/bin/env python
""" routines for reading halo data from cosmo sims.

These routines read halo data for simulations with subfind already run on them.
This serves as a wrapper to call readsnapHDF5 using offset and length values
obtained from the subfind catalog.

Example Usage:
import simread.readhaloHDF5 as readhalo
base = '/n/ghernquist/rmckinnon/Aq-C-5/output'
snapbase = "snapshot"
num = 255
h = readhalo.HaloReader(base, snapbase, num)
type = grpnr = subnr = 0
pos = h.read("POS ", type, grpnr, subnr)
"""

__author__ = "Mark Vogelsberger, Paul Torrey and contributing authors"
__copyright__ = "Copyright 2014, The Authors"
__credits__ = ["Mark Vogelsberger, Paul Torrey and contributing authors"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Paul Torrey"
__email__ = "ptorrey@mit.harvard.edu"
__status__ = "Beta -- forever."

import gc
import os
import sys
import numpy as np

import paul_illustris_tools.readsnapHDF5 as readsnapHDF5
import paul_illustris_tools.readsubfHDF5 as readsubfHDF5

#import readsnapHDF5 as readsnapHDF5
#import readsubHDF5 as readsubHDF5
#import simread.readsnapHDF5 as readsnapHDF5
#import simread.readsubfHDF5 as readsubfHDF5
import paul_illustris_tools.hdf5lib as hdf5lib
import paul_illustris_tools.naming as naming

#import hdf5lib
#import naming



class HaloReader(object):
    def __init__(self, base, num, long_ids=False, snapbase='snap',
                 double_output=False, verbose=False, run=None):
        self.base = base
        self.snapbase = snapbase
        self.num = num
        self.num_pad = str(num).zfill(3)
        self.verbose = verbose
        self.part_types = [0, 1, 3, 4, 5]
        keysel = ["ngroups", "nsubs", "GroupLenType", "GroupNsubs",
                  "GroupFirstSub", "SubhaloLenType"]
        self.cat = readsubfHDF5.subfind_catalog(base, num, long_ids=long_ids,
                                                double_output=double_output,
                                                keysel=keysel)
        if not hasattr(self.cat, "GroupLenType"):
            raise RuntimeError("Subfind catalog has no group or subhalo "
                               "information.")

        self.filenames = naming.get_snap_filenames( self.base, 
						    self.snapbase,
                                                    self.num)
        offsets = readsubfHDF5.get_offsets(self.cat, self.part_types,
                                           self.num, run)
        self.group_offset, self.halo_offset = offsets
        print(self.filenames)
        head = readsnapHDF5.snapshot_header(self.filenames[0])

        self.file_num = head.filenum
        assert(self.file_num == len(self.filenames))

        ntypes = 6
        self.file_type_numbers = np.zeros([self.file_num, ntypes],
                                          dtype="int64")
        cumcount = np.zeros(ntypes, dtype="int64")

        # Store in file_type_numbers[i, :] the cumulative number of particles
        # in all previous files.  Note we never need to open the last file.
        for i in range(0, self.file_num-1):
            if self.verbose:
                print("READHALO: initial read of file: %s" % self.filenames[i])
            head = readsnapHDF5.snapshot_header(self.filenames[i])

            cumcount[:] += head.npart[:]
            self.file_type_numbers[i+1, :] = cumcount[:]

    def read(self, block_name, parttype, fof_num, sub_num):
        if sub_num < 0 and fof_num < 0:
            # Load all of the non-FoF particles.
            off = (self.group_offset[-1, parttype] +
                   self.cat.GroupLenType[-1, parttype])
            left = 1e9                # reads the rest.
            print(off, left)

        if sub_num >= 0 and fof_num < 0:
            off = self.halo_offset[sub_num, parttype]
            left = self.cat.SubhaloLenType[sub_num, parttype]

        if fof_num >= 0 and sub_num < 0:
            off = self.group_offset[fof_num, parttype]
            left = self.cat.GroupLenType[fof_num, parttype]

        if sub_num >= 0 and fof_num >= 0:
            real_sub_num = sub_num + self.cat.GroupFirstSub[fof_num]
            off = self.halo_offset[real_sub_num, parttype]
            left = self.cat.SubhaloLenType[real_sub_num, parttype]

        if left == 0:
            if self.verbose:
                print("READHALO: no particles of type...returning")
            return

        # Get first file that contains particles of required halo/fof/etc.
        findex = np.argmax(self.file_type_numbers[:, parttype] > off) - 1
        # np.argmax returns 0 when the offset corresponds to a particle
        # in the last file.
        if findex == -1:
            findex = self.file_num - 1

        # Convert the overall offset to an offset for just the file given by
        # findex by subtracting off the number of particles in previous files.
        for fnr in range(0, findex):
            off -= (self.file_type_numbers[fnr+1, parttype] -
                    self.file_type_numbers[fnr, parttype])

        # Read data from file(s).
        first = True
        for fnr in range(findex, self.file_num):
            path = self.filenames[fnr]

            head = readsnapHDF5.snapshot_header(path)
            nloc = head.npart[parttype]

            if nloc > off:
                if self.verbose:
                    print("READHALO: data found in %s" % path)
                start = off
                if nloc - off > left:
                    # All remaining particles are in this file.
                    count = left
                else:
                    # Read to end of file.
                    count = nloc - off

                block = readsnapHDF5.read_block(path, block_name, parttype,
                                                slab_start=start,
                                                slab_len=count)
                if first:
                    data = block
                    first = False
                else:
                    data = np.append(data, block, axis=0)

                left -= count
                off += count
            if left == 0:
                break
            off -= nloc

        gc.collect()

        return data
