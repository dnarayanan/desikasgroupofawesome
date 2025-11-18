import numpy as np
import caesar,yt
import sys

snapshot = '/blue/narayanan/desika.narayanan/arepo_runs/cosmic_sands/m100/dm_boxes/m100n256/output/snapdir_019/snapshot_019.0.hdf5'
caesarfile = '/blue/narayanan/desika.narayanan/arepo_runs/cosmic_sands/m100/dm_boxes/m100n256/output/Groups/caesar_snapshot_019.hdf5'
icfile = '/blue/narayanan/desika.narayanan/arepo_runs/cosmic_sands/m100/dm_boxes/ICs/ics_m100n256.hdf5'
starthalo=1480
endhalo=1481



obj = caesar.load(caesarfile)
ic = icfile
ds = yt.load(snapshot)
ic_ds = yt.load(ic)
obj.yt_dataset = ds


for i in range(starthalo,endhalo):
    halonum = i
    print(halonum)
    outfile = 'run0_halo'+str(i)+'_mask.txt'
    obj.halos[halonum].write_IC_mask(ic_ds,outfile)#,radius_type='dm_r20',search_factor=0.5)
