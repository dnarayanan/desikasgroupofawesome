import h5py
import argparse
import sys, os
import traceback  # For detailed error tracebacks
import numpy as np
from glob import glob

# --- PATH SETUP ---
sys.path.insert(0, 'paul_illustris_tools_A1')

# --- CUSTOM IMPORTS ---
try:
    import paul_illustris_tools.readhaloHDF5 as readhalo
    # NEW: Import subfind reader explicitly to get properties readhalo skips
    import paul_illustris_tools.readsubfHDF5 as read_subf 
    print(f"DEBUG: Importing readhalo from: {readhalo.__file__}")
except Exception as e:
    print(f"FATAL ERROR: Could not import custom tools. Error: {e}")
    sys.exit(1)

# --- ARGUMENT PARSING ---
if len(sys.argv) < 2:
    print("Error: Missing command-line argument for base_path.")
    print("Usage: python filter_illustris_many.py /path/to/simulation/output/")
    sys.exit(1)

base_path = sys.argv[1]
ZOOM = True
DEBUG = False 
startsnap = 1
endsnap = 600
snaprange = np.arange(startsnap, endsnap)

# --- MAIN LOOP ---
for snap_num in snaprange:
    print('halo reader for snap:', snap_num)
    
    # 1. Initialize Halo Reader (gets offsets/lengths)
    try:
        h = readhalo.HaloReader(str(base_path), '0'+str(snap_num), int(snap_num))
    except Exception as e:
        print(f"Could not read halo catalog for snapshot {snap_num}. Skipping.")
        print(f"Error (short): {e}")
        print("------- FULL TRACEBACK -------")
        traceback.print_exc() 
        print("------------------------------")
        continue
    
    print('getting offsets')
    
    if ZOOM == False:
        num_galaxies = h.halo_offset.shape[0]
    else:
        num_galaxies = 1 # Only process the central zoom galaxy

    output_dir = base_path+'/snapdir_'+str(snap_num).zfill(3)+'/filtered_snaps/'
    print("writing to output directory: "+output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(output_dir+" does not exist: creating now")

    for gal in range(num_galaxies):
        
        print("working on galaxy: ", gal)
        outfile = output_dir+'/snapshot_'+str(snap_num).zfill(3)+'.gal'+str(gal)+'.hdf5'
        print("outfile: ", outfile)

        outfile_zoom =  output_dir+'/snapshot_'+str(snap_num).zfill(3)+'.hdf5'
        
        # --- EXTRACT OFFSETS AND LENGTHS ---
        # PartType0 = Gas, PartType1 = DM, PartType3 = Dust/Tracers, PartType4 = Stars
        try:
            gas_offset = h.halo_offset[gal][0]
            dm_offset = h.halo_offset[gal][1]
            dust_offset = h.halo_offset[gal][3]
            star_offset = h.halo_offset[gal][4]
            
            gas_len = h.cat.SubhaloLenType[gal][0]
            dm_len = h.cat.SubhaloLenType[gal][1]
            star_len = h.cat.SubhaloLenType[gal][4]
            dust_len = h.cat.SubhaloLenType[gal][3]
        except Exception as e:
             print(f"Error reading offsets for galaxy {gal}: {e}")
             continue
        
        if DEBUG:
            print('gas offset:',gas_offset, 'gas len:',gas_len)
            print('dm offset:',dm_offset, 'dm len:',dm_len)
            print('star offset:',star_offset, 'star len:',star_len)
            print('dust offset:',dust_offset, 'dust len:',dust_len)
            
        
        files_dir = base_path+'/snapdir_'+str(snap_num).zfill(3)+'/'
        files = sorted(glob(files_dir+'*.hdf5'), key = lambda name: int(name.split('.')[-2]))

        this_file_start0 = 0
        this_file_start1 = 0
        this_file_start3 = 0
        this_file_start4 = 0

        # --- Gas Particles (PartType0) ---
        if gas_len > 0:
            if DEBUG: print('starting with gas particles')
            wrote_gas_header = False
            for snap_file in files:
                with h5py.File(snap_file, 'r') as input_file:
                    if 'PartType0' not in input_file: continue
                    this_file_end0 = this_file_start0 + len(input_file['PartType0']['Coordinates'])
                    if (gas_offset >= this_file_end0):
                        this_file_start0 = this_file_end0
                        continue
                    if (this_file_start0 > (gas_offset + gas_len)): break
                    if not wrote_gas_header:
                        with h5py.File(outfile,"w") as output_file:
                            output_file.copy(input_file['Header'], 'Header')
                            output_file.copy(input_file['Config'], 'Config')
                            output_file.create_group('PartType0')
                        wrote_gas_header = True
                    with h5py.File(outfile, "a") as output_file:
                        if DEBUG: print('writing gas from file',snap_file)
                        for k in input_file['PartType0']:
                            start_idx = max(0, gas_offset - this_file_start0)
                            end_idx = min(len(input_file['PartType0'][k]), (gas_offset + gas_len) - this_file_start0)
                            in_data = input_file['PartType0'][k][start_idx:end_idx]
                            if len(in_data) == 0: continue
                            if k in output_file['PartType0']:
                                if DEBUG: print('[gas] appending data for key:', k)
                                old_data = output_file['PartType0'][k][()]
                                new_data = np.concatenate([old_data, in_data], axis=0)
                                del output_file['PartType0'][k]
                                output_file['PartType0'].create_dataset(k, data=new_data)
                            else:
                                if DEBUG: print('[gas] writing new data for key:', k)
                                output_file['PartType0'].create_dataset(k, data=in_data, maxshape=(None,) + in_data.shape[1:])
                this_file_start0 = this_file_end0

        # --- Dark Matter Particles (PartType1) ---
        if dm_len > 0:
            if DEBUG: print('now doing DM particles')
            for snap_file in files:
                with h5py.File(snap_file, 'r') as input_file:
                    if 'PartType1' not in input_file: continue
                    
                    this_file_end1 = this_file_start1 + len(input_file['PartType1']['Coordinates'])
                    
                    if (dm_offset >= this_file_end1):
                        this_file_start1 = this_file_end1
                        continue
                    if (this_file_start1 > (dm_offset + dm_len)): break
                    
                    with h5py.File(outfile,"a") as output_file:
                        if 'Header' not in output_file:
                             output_file.copy(input_file['Header'], 'Header')
                             output_file.copy(input_file['Config'], 'Config')
                        
                        if 'PartType1' not in output_file: output_file.create_group('PartType1')
                        if DEBUG: print('writing DM from file', snap_file)
                        
                        for k in input_file['PartType1']:
                            start_idx = max(0, dm_offset - this_file_start1)
                            end_idx = min(len(input_file['PartType1'][k]), (dm_offset + dm_len) - this_file_start1)
                            in_data = input_file['PartType1'][k][start_idx:end_idx]
                            
                            if len(in_data) == 0: continue
                            
                            if k in output_file['PartType1']:
                                if DEBUG: print('[dm] appending data for key:', k)
                                old_data = output_file['PartType1'][k][()]
                                new_data = np.concatenate([old_data, in_data], axis=0)
                                del output_file['PartType1'][k]
                                output_file['PartType1'].create_dataset(k, data=new_data)
                            else:
                                if DEBUG: print('[dm] writing new data for key:', k)
                                output_file['PartType1'].create_dataset(k, data=in_data, maxshape=(None,) + in_data.shape[1:])
                this_file_start1 = this_file_end1

        # --- Dust Particles (PartType3) ---
        if dust_len > 0:
            if DEBUG: print('now doing dust particles')
            for snap_file in files:
                with h5py.File(snap_file, 'r') as input_file:
                    if 'PartType3' not in input_file: continue
                    this_file_end3 = this_file_start3 + len(input_file['PartType3']['Coordinates'])
                    if (dust_offset >= this_file_end3):
                        this_file_start3 = this_file_end3
                        continue
                    if (this_file_start3 > (dust_offset + dust_len)): break
                    with h5py.File(outfile,"a") as output_file:
                        if 'PartType3' not in output_file: output_file.create_group('PartType3')
                        if DEBUG: print('writing dust from file', snap_file)
                        for k in input_file['PartType3']:
                            start_idx = max(0, dust_offset - this_file_start3)
                            end_idx = min(len(input_file['PartType3'][k]), (dust_offset + dust_len) - this_file_start3)
                            in_data = input_file['PartType3'][k][start_idx:end_idx]
                            if len(in_data) == 0: continue
                            if k in output_file['PartType3']:
                                if DEBUG: print('[dust] appending data for key:', k)
                                old_data = output_file['PartType3'][k][()]
                                new_data = np.concatenate([old_data, in_data], axis=0)
                                del output_file['PartType3'][k]
                                output_file['PartType3'].create_dataset(k, data=new_data)
                            else:
                                if DEBUG: print('[dust] writing new data for key:', k)
                                output_file['PartType3'].create_dataset(k, data=in_data, maxshape=(None,) + in_data.shape[1:])
                this_file_start3 = this_file_end3

        # --- Star Particles (PartType4) ---
        if star_len > 0:
            if DEBUG: print('now doing star particles')
            for snap_file in files:
                with h5py.File(snap_file, 'r') as input_file:
                    if 'PartType4' not in input_file: continue
                    this_file_end4 = this_file_start4 + len(input_file['PartType4']['Coordinates'])
                    if (star_offset >= this_file_end4):
                        this_file_start4 = this_file_end4
                        continue
                    if (this_file_start4 > (star_offset + star_len)): break
                    with h5py.File(outfile,"a") as output_file:
                        if 'PartType4' not in output_file: output_file.create_group('PartType4')
                        if DEBUG: print('writing stars from file', snap_file)
                        for k in input_file['PartType4']:
                            start_idx = max(0, star_offset - this_file_start4)
                            end_idx = min(len(input_file['PartType4'][k]), (star_offset + star_len) - this_file_start4)
                            in_data = input_file['PartType4'][k][start_idx:end_idx]
                            if len(in_data) == 0: continue
                            if k in output_file['PartType4']:
                                if DEBUG: print('[stars] appending data for key:', k)
                                old_data = output_file['PartType4'][k][()]
                                new_data = np.concatenate([old_data, in_data], axis=0)
                                del output_file['PartType4'][k]
                                output_file['PartType4'].create_dataset(k, data=new_data)
                            else:
                                if DEBUG: print('[stars] writing new data for key:', k)
                                output_file['PartType4'].create_dataset(k, data=in_data, maxshape=(None,) + in_data.shape[1:])
                this_file_start4 = this_file_end4

        # ======================================================================
        # FINAL STEP: SAVE DIAGNOSTICS AND UPDATE HEADER
        # ======================================================================
        print("Updating header and saving diagnostics...")
        
        try:
            # --- NEW: Explicitly load the subfind catalog with the keys we need ---
            # We do this separately from 'h' to ensure we get exactly the fields we want.
            needed_keys = [
                'SubhaloGrNr', 'GroupFirstSub', 'Group_M_Crit200', 'Group_R_Crit200',
                'SubhaloPos', 'SubhaloVel', 'SubhaloVmax', 'SubhaloVmaxRad',
                'SubhaloSpin', 'SubhaloHalfmassRadType'
            ]
            
            # Load catalog for this snapshot
            # Note: subfind_catalog expects the BASE directory and the SNAP NUMBER
            full_cat = read_subf.subfind_catalog(base_path, snap_num, keysel=needed_keys)
            
            with h5py.File(outfile, "a") as f:
                # --- A. Update Particle Counts ---
                npart = np.zeros(6, dtype=np.uint32)
                if 'PartType0' in f: npart[0] = len(f['PartType0']['Coordinates'])
                if 'PartType1' in f: npart[1] = len(f['PartType1']['Coordinates'])
                if 'PartType3' in f: npart[3] = len(f['PartType3']['Coordinates'])
                if 'PartType4' in f: npart[4] = len(f['PartType4']['Coordinates'])
                
                if 'NumPart_ThisFile' in f['Header'].attrs:
                    f['Header'].attrs.modify('NumPart_ThisFile', npart)
                    f['Header'].attrs.modify('NumPart_Total', npart)
                else:
                    f['Header'].attrs['NumPart_ThisFile'] = npart
                    f['Header'].attrs['NumPart_Total'] = npart

                f['Header'].attrs['NumFilesPerSnapshot'] = 1
                f['Header'].attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype=np.uint32)

                # --- B. Save Subfind Physical Properties ---
                # 1. Identify Parent Group info
                # Note: gal is 0 in zoom simulations. We trust that readsubf catalog index 0 
                # corresponds to readhalo offset 0.
                parent_group_idx = full_cat.SubhaloGrNr[gal]
                
                # 2. Group Properties
                try:
                    m200 = float(full_cat.Group_M_Crit200[parent_group_idx])
                    r200 = float(full_cat.Group_R_Crit200[parent_group_idx])
                    first_sub = int(full_cat.GroupFirstSub[parent_group_idx])
                except Exception as e:
                    print(f"Warning: Group lookup failed ({e}). Setting to -1.")
                    m200 = -1.0; r200 = -1.0; first_sub = -1

                # 3. Subhalo Properties
                sub_pos = full_cat.SubhaloPos[gal] 
                sub_vel = full_cat.SubhaloVel[gal]
                sub_vmax = float(full_cat.SubhaloVmax[gal])
                sub_vmax_rad = float(full_cat.SubhaloVmaxRad[gal])
                sub_spin = full_cat.SubhaloSpin[gal]
                sub_rhalf = full_cat.SubhaloHalfmassRadType[gal] 
                
                # 4. Is Central Check
                is_central = 1 if (gal == 0 and first_sub == 0) else 0

                # 5. Write Attributes
                head = f['Header'].attrs
                head['Group_M_Crit200'] = m200
                head['Group_R_Crit200'] = r200
                head['GroupFirstSub'] = first_sub
                head['SubhaloGrNr'] = parent_group_idx
                head['IsCentral'] = is_central
                
                head['SubhaloPos'] = sub_pos
                head['SubhaloVel'] = sub_vel
                head['SubhaloVmax'] = sub_vmax
                head['SubhaloVmaxRad'] = sub_vmax_rad
                head['SubhaloSpin'] = sub_spin
                head['SubhaloHalfmassRadType'] = sub_rhalf

        except Exception as e:
            print(f"Could not update header/diagnostics for {outfile}.")
            print(f"Error: {e}")
            traceback.print_exc()

        if ZOOM:
            if os.path.lexists(outfile_zoom): os.remove(outfile_zoom)
            source_file_name = os.path.basename(outfile)
            os.symlink(source_file_name, outfile_zoom)
            print(f"Created symlink: {outfile_zoom} -> {source_file_name}")

print("Processing complete.")
