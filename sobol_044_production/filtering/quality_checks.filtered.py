import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py
import os
import sys
from astropy.io import fits
import multiprocessing
from tqdm import tqdm

# ==============================================================================
#           SETUP: PATHS AND CONFIGURATION
# ==============================================================================
print("Initializing analysis script...")

# --- CORE PATHS ---
sys.path.insert(0, '/home/desika.narayanan/torreylabtools_A1/')

PLOT_OUTPUT_DIR = './plots'
SIM_LIST_FILE = 'sim_list.txt'

# --- PARALLELISM CONTROL ---
NUM_PROCESSES = 8 

# --- SIMULATION PARAMETERS ---
START_SNAP = 1
END_SNAP = 600

# --- OBSERVATIONAL DATA PATHS ---
SHIVAEI_DATA_FILE = './datafiles/shivaei.txt'
SHIM_FITS_FILE = './datafiles/shim.fits'
BEHROOZI_DIR = '/home/desika.narayanan/behroozi_abundance_matching/release-sfh_z0_z8_052913/smmr/'

# --- PLOT STYLING PREFERENCES ---
plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.titlesize': 20
})

# --- Import custom tools ---
try:
    from live_dust_util import SnapshotContainer as Snap
    from live_dust_util import GrainSizeDistribution as GSD
except ImportError as e:
    print(f"FATAL ERROR: Could not import live_dust_util. Check your sys.path. Error: {e}")
    sys.exit()

# ==============================================================================
#                   1. DATA LOADING FUNCTIONS
# ==============================================================================

def scrape_param_file(run_path):
    """Reads the simulation parameter file for annotation."""
    param_file = os.path.join(run_path, 'param.txt')
    keys_to_find = {"DensThreshold", "ThermalFeedbackEfficiency", "SfrEfficiency", 
                    "DLClumpingFac", "InputTimeMomRadiationFeedback", "LumToMassRatioRadiationFeedback"}
    params = {}
    try:
        with open(param_file, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                if parts[0] in keys_to_find:
                    params[parts[0]] = parts[1]
    except FileNotFoundError as e:
        print(f"  Warning: param.txt not found in {run_path}. Error: {e}")
    return params

def load_observational_data():
    """Loads external observational datasets for comparison."""
    print("Loading observational data...")
    obs_data = {}
    
    # Load Shivaei+20
    try:
        shivaei_d = np.genfromtxt(SHIVAEI_DATA_FILE, skip_header=6)
        obs_data['shivaei'] = {'qpah': np.asarray(shivaei_d[:, 1]) * 0.01, '12logoh': shivaei_d[:, 5]}
    except Exception as e:
        print(f"  Warning: Error processing SHIVAEI_DATA_FILE. Error: {e}")
        obs_data['shivaei'] = None
        
    # Load Shim+21
    try:
        with fits.open(SHIM_FITS_FILE, memmap=True) as shim_hdu:
            shim_d = shim_hdu[1].data
            obs_data['shim'] = {'qpah': np.asarray(shim_d['qpah']) * 0.01, '12logoh': shim_d['12+log(O/H)']}
    except Exception as e:
        print(f"  Warning: Error processing SHIM_FITS_FILE. Error: {e}")
        obs_data['shim'] = None
    return obs_data

def extract_simulation_data_for_run(run_path):
    """
    Extracts physical properties DIRECTLY from filtered snapshots.
    Does not rely on external Subfind catalogs.
    """
    output_path = os.path.join(run_path, 'output/')
    run_name = os.path.basename(os.path.normpath(run_path))
    
    props = {
        'redshift': [], 'mstar': [], 'mdust': [], 'mgas': [], 'mhalo': [],
        'oh12': [], 'qpah': [], 'pos': [], 'snap_num': [], 'vmax': [],
        'f_carbonaceous': [], 'stl_ratio': []
    }
    
    if not os.path.isdir(output_path):
        print(f"  Warning: 'output' directory not found in {run_path}. Skipping.")
        return props

    for snap_num in range(START_SNAP, END_SNAP):
        # PATH: Logic for filtered snapshots (symlinked as snapshot_XXX.hdf5 in filtered dir)
        snap_dir = os.path.join(output_path, f'snapdir_{snap_num:03d}')
        snap_file = os.path.join(snap_dir, 'filtered_snaps', f'snapshot_{snap_num:03d}.hdf5')
        
        if not os.path.exists(snap_file): continue
        
        try:
            data_points = {}
            
            with h5py.File(snap_file, 'r') as f:
                header = f['Header'].attrs
                
                # --- 1. Time / Redshift ---
                a = header['Time']
                redshift = (1./a) - 1 if a > 0 else 0
                data_points['redshift'] = redshift
                data_points['snap_num'] = snap_num

                # --- 2. Cosmology & Units ---
                # Arepo masses are 1e10 Msun/h. Convert to Msun.
                h_hubble = header.get('HubbleParam', 0.7)
                mass_conversion = 1e10 / h_hubble

                # --- 3. Read Subfind Props from Header (Saved by Filtering Script) ---
                # Note: We default to -1 if keys are missing (e.g. if older filtering script was used)
                data_points['mhalo'] = header.get('Group_M_Crit200', -1.0) * mass_conversion
                data_points['vmax']  = header.get('SubhaloVmax', -1.0)
                
                # Center of Potential (Critical for image centering)
                if 'SubhaloPos' in header:
                    data_points['pos'] = header['SubhaloPos']
                else:
                    # Fallback to gas COM if SubhaloPos is missing
                    if 'PartType0' in f:
                        data_points['pos'] = np.mean(f['PartType0']['Coordinates'], axis=0)
                    else:
                        data_points['pos'] = np.array([0., 0., 0.])

                # --- 4. Sum Particle Masses ---
                # Stellar Mass
                if 'PartType4' in f and 'Masses' in f['PartType4']:
                    data_points['mstar'] = np.sum(f['PartType4']['Masses']) * mass_conversion
                else:
                    data_points['mstar'] = 1e-10 # Avoid log(0)

                # Gas Mass
                if 'PartType0' in f and 'Masses' in f['PartType0']:
                    data_points['mgas'] = np.sum(f['PartType0']['Masses']) * mass_conversion
                else:
                    data_points['mgas'] = 1e-10

                # Dust Mass
                if 'PartType3' in f and 'Masses' in f['PartType3']:
                    data_points['mdust'] = np.sum(f['PartType3']['Masses']) * mass_conversion
                else:
                    data_points['mdust'] = 1e-10

                # --- 5. Calculate Metallicity (SFR Weighted) ---
                if 'PartType0' in f and 'GFM_Metallicity' in f['PartType0'] and 'StarFormationRate' in f['PartType0']:
                    gas_z = f['PartType0']['GFM_Metallicity'][:]
                    sfr = f['PartType0']['StarFormationRate'][:]
                    
                    sf_mask = sfr > 0
                    if np.any(sf_mask):
                        w_mean_z = np.average(gas_z[sf_mask], weights=sfr[sf_mask])
                        # Solar abundance approx (Asplund)
                        data_points['oh12'] = np.log10(w_mean_z * 0.5 / (0.70 * 16.0)) + 12.0
                    else:
                        data_points['oh12'] = -1
                else:
                    data_points['oh12'] = -1

            # --- 6. Live Dust Tools ---
            # Point to the filtered directory where the snapshot lives
            filtered_snap_path = os.path.dirname(snap_file) + '/'
            
            try:
                gsd_snapshot = Snap(snap_num, filtered_snap_path)
                gsd = GSD(gsd_snapshot, a=10.**np.linspace(-3, 0, 16))
                data_points['qpah'], _, _ = gsd.compute_q_pah(size=3.e-3)
                abundances = gsd.compute_abundances()
                data_points['f_carbonaceous'] = abundances.get('Aliphatic C', 0) + abundances.get('PAH', 0)
                stl = gsd.compute_small_to_large_ratio(size=3.e-3)
                data_points['stl_ratio'] = stl.get('Silicate',0) + stl.get('Carbonaceous',0) + stl.get('PAH',0)
            except Exception as e:
                # print(f"    (Dust calc failed snap {snap_num}: {e})")
                data_points['qpah'] = -1
                data_points['f_carbonaceous'] = -1
                data_points['stl_ratio'] = -1

            # Append to arrays
            for key in props.keys():
                props[key].append(data_points.get(key, -1))

        except Exception as e:
            print(f"  Warning: Failed to read filtered snap {snap_num} for {run_name}. Error: {e}", flush=True)
            for key in props.keys(): 
                props[key].append(np.array([-1., -1., -1.]) if key == 'pos' else -1)
            
    for key in props: props[key] = np.array(props[key])
    return props

# ==============================================================================
#                   2. PLOTTING FUNCTIONS
# ==============================================================================

def plot_mzr_fit(ax):
    log_mstar = np.linspace(8.5, 11.5, 100)
    oh12 = -1.492 + 1.847 * log_mstar - 0.08026 * (log_mstar**2)
    ax.plot(10**log_mstar, oh12, color='limegreen', linestyle='-.', lw=3, zorder=5, label='MZR Fit (z~0)')

def plot_lelli_2019_btfr(ax):
    log_mbary = np.linspace(7, 12, 100)
    log_vflat = (log_mbary - 1.78) / 4.00
    ax.plot(10**log_mbary, 10**log_vflat, color='crimson', linestyle='--', lw=3, zorder=5, label='Lelli+19 BTFR (z=0)')

def create_galaxy_summary_plot(sim_data, obs_data, run_params, output_filename, run_name, run_path):
    import yt
    # Suppress yt logging
    yt.funcs.mylog.setLevel(50)
    
    plt.style.use('seaborn-v0_8-talk')
    fig, axes = plt.subplots(5, 3, figsize=(20, 32))

    fig.suptitle(f"Galaxy Evolution Summary: {run_name}", fontsize=28, y=1.0)
    param_text = (f"DensThresh: {run_params.get('DensThreshold', 'N/A')} | SfrEff: {run_params.get('SfrEfficiency', 'N/A')}\n"
                  f"ThermFdbkEff: {run_params.get('ThermalFeedbackEfficiency', 'N/A')} | DLCF: {run_params.get('DLClumpingFac', 'N/A')}\n"
                  f"InpTimeMomRad: {run_params.get('InputTimeMomRadiationFeedback', 'N/A')} | LumToMass: {run_params.get('LumToMassRatioRadiationFeedback', 'N/A')}")
    fig.text(0.5, 0.965, param_text, ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='k', lw=1, alpha=0.9))

    target_zs = [6, 2, 0]
    all_redshifts, all_mstar, all_mhalo = sim_data['redshift'], sim_data['mstar'], sim_data['mhalo']
    all_mgas, all_mdust, all_vmax = sim_data['mgas'], sim_data['mdust'], sim_data['vmax']
    all_oh12, all_qpah = sim_data['oh12'], sim_data['qpah']
    all_pos, all_snap_nums = sim_data['pos'], sim_data['snap_num']
    
    for col, z_target in enumerate(target_zs):
        ax_smhm, ax_mzr, ax_proj = axes[0, col], axes[1, col], axes[3, col]
        if col > 0: ax_smhm.set_yticklabels([]); ax_mzr.set_yticklabels([]); ax_proj.set_yticklabels([]); ax_proj.set_xticklabels([])
        
        valid_indices = np.where(all_redshifts > -1)[0]
        if not valid_indices.size:
            for ax in [ax_smhm, ax_mzr, ax_proj]: ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
            continue
            
        diff = np.abs(all_redshifts[valid_indices] - z_target)
        if np.min(diff) < 0.5:
            idx = valid_indices[np.argmin(diff)]
            actual_z = all_redshifts[idx]; snap_num = int(all_snap_nums[idx])
            
            # SMHM Plot
            if all_mstar[idx] > 0 and all_mhalo[idx] > 0: 
                ax_smhm.scatter(all_mhalo[idx], all_mstar[idx], color='dodgerblue', s=200, edgecolor='k', zorder=15, label='This Work')
            
            # MZR Plot
            if all_mstar[idx] > 0 and all_oh12[idx] > 0: 
                ax_mzr.scatter(all_mstar[idx], all_oh12[idx], color='rebeccapurple', s=200, edgecolor='k', zorder=15, label='This Work')
            
            ax_smhm.set_title(f"z = {actual_z:.2f}", fontsize=20)
            
            # YT Projection
            try:
                # Use the filtered snapshot directly
                snap_path = os.path.join(run_path, 'output', f'snapdir_{snap_num:03d}', 'filtered_snaps', f'snapshot_{snap_num:03d}.hdf5')
                
                ds = yt.load(snap_path)
                # Center on the 'pos' we read from the header (SubhaloPos from Potential Min)
                proj = ds.proj(('gas', 'density'), 'x', center=all_pos[idx])
                frb = proj.to_frb(width=(10, 'kpc'), resolution=800)
                gas_density_data = np.array(frb[('gas', 'density')])
                
                positive_data = gas_density_data[gas_density_data > 0]
                norm = LogNorm(vmin=positive_data.min(), vmax=positive_data.max()) if positive_data.any() else LogNorm()
                
                # ADDED: origin='lower' to fix orientation
                ax_proj.imshow(gas_density_data, norm=norm, cmap='inferno', origin='lower')
                
                ax_proj.text(0.05, 0.95, f'z = {actual_z:.2f}', transform=ax_proj.transAxes, color='white', fontsize=14, ha='left', va='top')
                ax_proj.set_xticks([]); ax_proj.set_yticks([])
            except Exception as e:
                ax_proj.text(0.5, 0.5, 'Image Failed', ha='center', va='center', color='white', transform=ax_proj.transAxes)
                print(f"  Warning: yt image projection failed for {run_name}, snap {snap_num}. Error: {e}", flush=True)
        else:
            ax_smhm.set_title(f"z ~ {z_target} (No Data)", fontsize=20)
            # FIX: Use transform=ax.transAxes to keep text in the box. Remove ticks.
            for ax in [ax_smhm, ax_mzr, ax_proj]: 
                ax.text(0.5, 0.5, 'No Snapshot Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
        
        # Behroozi Lines
        try:
            z_file_map = {6: '7.00', 2: '2.00', 0: '0.10', 1: '1.00'}; z_str = z_file_map.get(z_target, '7.00')
            behroozi_data = np.loadtxt(os.path.join(BEHROOZI_DIR, f'c_smmr_z{z_str}_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat'))
            mhalo_b, smhm_ratio = 10.**(behroozi_data[:,0]), 10.**(behroozi_data[:,1])
            ax_smhm.plot(mhalo_b, smhm_ratio * mhalo_b, color='crimson', lw=2.5, label=f'Behroozi+13 (z~{int(np.round(float(z_str)))})', zorder=10)
        except Exception:
            pass
            
        ax_smhm.set_xscale('log'); ax_smhm.set_yscale('log')
        ax_smhm.set_xlabel(r'Halo Mass ($M_\odot$)', fontsize=16)
        if col == 0: ax_smhm.legend(loc='lower right', fontsize=14)
        
        # Reference Data for MZR
        if z_target == 6:
            heintz_mstar, heintz_12logOH = [8.84,8.19,8.72,8.88,8.45,8.33,8.66,8.40,10.0,9.05,9.07,9.47,9.04,8.64], [7.56,7.29,7.68,7.97,7.82,7.49,7.42,7.86,8.06,7.55,7.75,8.00,7.62,7.42]
            ax_mzr.scatter(10.**np.asarray(heintz_mstar), heintz_12logOH, facecolor='#FF3333', marker='s', s=150, edgecolor='black', linewidth=1.5, label='Heintz+23 (z~7-8)', zorder=10)
        if z_target == 2:
            erb_log_mstar = np.linspace(9.0, 11.0, 10); erb_12logOH = 8.7 + 0.23 * (erb_log_mstar - 10)
            ax_mzr.plot(10**erb_log_mstar, erb_12logOH, color='dodgerblue', label='Erb et al. 2006 (z=2)')
        if z_target == 0:
            plot_mzr_fit(ax_mzr)
            
        ax_mzr.set_xscale('log')
        ax_mzr.set_xlabel(r'Stellar Mass ($M_\odot$)', fontsize=16)
        if col == 0: ax_mzr.legend(loc='lower right', fontsize=14)

    # Row 3: Dust Properties
    ax_pah_z, ax_gdr, ax_dts = axes[2, 0], axes[2, 1], axes[2, 2]
    valid = (all_qpah > 1e-17) & (all_oh12 > 0)
    if np.any(valid):
        sc = ax_pah_z.scatter(all_oh12[valid], all_qpah[valid], c=all_redshifts[valid], cmap='viridis', s=50, zorder=15)
        cbar = fig.colorbar(sc, ax=ax_pah_z, label='Redshift (z)')
        cbar.ax.set_ylabel('Redshift (z)', fontsize=16)
        
    if obs_data.get('shim'): ax_pah_z.scatter(obs_data['shim']['12logoh'], obs_data['shim']['qpah'], marker='*', alpha=0.25, label='Shim+21', zorder=10, s=100, color='gray')
    if obs_data.get('shivaei'): ax_pah_z.scatter(obs_data['shivaei']['12logoh'], obs_data['shivaei']['qpah'], marker='P', alpha=0.5, label='Shivaei+20', zorder=10, s=100, color='silver')
    ax_pah_z.set_yscale('log'); ax_pah_z.set_xlim([6.5,9.5]); ax_pah_z.set_ylim([1.e-5,1])
    ax_pah_z.set_xlabel(r'12+log(O/H)', fontsize=16); ax_pah_z.set_ylabel(r'q$_\mathrm{PAH}$', fontsize=16)
    ax_pah_z.legend(fontsize=14)
    
    ax_gdr.set_yticklabels([]); valid = (all_mgas > 0) & (all_mdust > 0) & (all_oh12 > 0)
    if np.any(valid):
        gdr = all_mgas[valid] / all_mdust[valid]
        sc2 = ax_gdr.scatter(all_oh12[valid], gdr, c=all_redshifts[valid], cmap='viridis', s=50, zorder=15)
        cbar2 = fig.colorbar(sc2, ax=ax_gdr, label='Redshift (z)')
        cbar2.ax.set_ylabel('Redshift (z)', fontsize=16)
        
    x_array = np.linspace(6.5, 9.5, 10); y_rr = 2.21 + (8.69 - x_array); ax_gdr.plot(x_array, 10.**y_rr, color='gray', linestyle='--', label='Remy-Ruyer+14 (z~0)')
    
    # --- RESTORED DE VIS+19 LINES ---
    a_devis = [2.45, 2, 2.13, 2.15, 2.1, 1.78, 1.95]
    b_devis = [-23.3, -19.56, -20.93, -21.19, -20.91, -18.52, -19.96]
    for i in range(len(a_devis)): 
        log_md_mg_devis = a_devis[i] * x_array + b_devis[i]
        md_mg_devis = 10.**(log_md_mg_devis)
        mg_md_devis = 1. / md_mg_devis
        ax_gdr.plot(x_array, mg_md_devis, color='darkorange', linestyle=':', label='De Vis+19 (z~0)' if i==0 else "")

    ax_gdr.set_yscale('log'); ax_gdr.set_xlim([6.5,9.5]); ax_gdr.set_ylim([10, 1e5])
    ax_gdr.set_xlabel(r'12+log(O/H)', fontsize=16); ax_gdr.set_ylabel(r'$M_\mathrm{gas} / M_\mathrm{dust}$', fontsize=16)
    ax_gdr.legend(fontsize=14)
    
    ax_dts.set_yticklabels([]); valid = (all_mstar > 0) & (all_mdust > 0)
    if np.any(valid):
        sc3 = ax_dts.scatter(all_mstar[valid], all_mdust[valid], c=all_redshifts[valid], cmap='viridis', s=50, zorder=15)
        cbar3 = fig.colorbar(sc3, ax=ax_dts, label='Redshift (z)')
        cbar3.ax.set_ylabel('Redshift (z)', fontsize=16)
        
    xarray_log = np.linspace(7, 12, 100); xarray = 10.**xarray_log
    for ratio_exp in [-2, -3, -4]: ax_dts.plot(xarray, xarray * 10**ratio_exp, color='darkgray', ls=':', alpha=0.9, zorder=1); ax_dts.annotate(fr'$M_\mathrm{{d}}/M_* = 10^{{{ratio_exp}}}$', xy=(1e10, 1e10 * 10**ratio_exp), rotation=35, fontsize=11, color='dimgray')
    ax_dts.set_xscale('log'); ax_dts.set_yscale('log'); ax_dts.set_xlim([1e7, 5e11]); ax_dts.set_ylim([1e3, 1e9])
    ax_dts.set_xlabel(r'$M_* \ (M_\odot)$', fontsize=16); ax_dts.set_ylabel(r'$M_\mathrm{dust} \ (M_\odot)$', fontsize=16)

    # Row 5: BTFR (Vmax vs Baryonic Mass)
    gs = axes[4, 0].get_gridspec()
    for ax in axes[4, :]: ax.remove()
    ax_vmax = fig.add_subplot(gs[4, :])
    baryonic_mass = all_mstar + all_mgas; valid = (baryonic_mass > 0) & (all_vmax > 0)
    if np.any(valid):
        sc = ax_vmax.scatter(baryonic_mass[valid], all_vmax[valid], c=all_redshifts[valid], cmap='viridis', s=50, alpha=0.7, zorder=15)
        cbar4 = fig.colorbar(sc, ax=ax_vmax, label='Redshift (z)')
        cbar4.ax.set_ylabel('Redshift (z)', fontsize=16)
        
    plot_lelli_2019_btfr(ax_vmax)
    ax_vmax.set_title(r'Baryonic Tully-Fisher Relation ($V_{max}$ vs. $M_{baryon}$)', fontsize=20)
    ax_vmax.set_xlabel(r'Baryonic Mass ($M_*+M_{gas}$) [$M_\odot$]', fontsize=16)
    ax_vmax.set_ylabel(r'$V_{max}$ (km/s)', fontsize=16)
    ax_vmax.set_xscale('log'); ax_vmax.set_yscale('log'); ax_vmax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_vmax.legend(fontsize=14)

    axes[0, 0].set_ylabel(r'Stellar Mass ($M_\odot$)', fontsize=16)
    axes[1, 0].set_ylabel(r'12 + log(O/H)', fontsize=16)
    axes[0, 0].set_xlim(1e9, 1e14); axes[0, 0].set_ylim(1e6, 1e12); axes[1, 0].set_ylim(6.8, 9.2)
    
    # FIX: Standard layout (bbox_inches removed)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_filename, dpi=300) 
    plt.close(fig)
    print(f"  -> Saved summary plot: {os.path.basename(output_filename)}")

def create_dust_diagnostics_plot(sim_data, obs_data, run_params, output_filename, run_name):
    """Placeholder for the second plot type if needed."""
    pass

# ==============================================================================
#           3. WORKER FUNCTION AND PARALLEL MAIN EXECUTION BLOCK
# ==============================================================================

obs_data_global = None

def init_worker(obs_data_from_main):
    global obs_data_global
    obs_data_global = obs_data_from_main

def process_single_run(run_path):
    try:
        run_name = os.path.basename(os.path.normpath(run_path))
        print(f"Processing run: {run_name}")
        
        run_params = scrape_param_file(run_path)
        sim_data = extract_simulation_data_for_run(run_path)
        
        if np.sum(sim_data['redshift'] > -1) == 0:
            print(f"  -> No valid snapshots found for {run_name}. Skipping.")
            return

        summary_plot_file = os.path.join(PLOT_OUTPUT_DIR, f"{run_name}_1_summary_evolution.png")
        diagnostics_plot_file = os.path.join(PLOT_OUTPUT_DIR, f"{run_name}_2_dust_diagnostics.png")

        create_galaxy_summary_plot(sim_data, obs_data_global, run_params, summary_plot_file, run_name, run_path)
        create_dust_diagnostics_plot(sim_data, obs_data_global, run_params, diagnostics_plot_file, run_name)
    except Exception as e:
        print(f"!!! ERROR processing {os.path.basename(os.path.normpath(run_path))}: {e}", flush=True)

def main():
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True); print(f"Plots will be saved in: {PLOT_OUTPUT_DIR}")
    obs_data = load_observational_data()
    
    run_paths = []
    if not os.path.exists(SIM_LIST_FILE):
        print(f"FATAL ERROR: Simulation list file not found: {SIM_LIST_FILE}"); return
        
    print(f"Loading simulation paths from: {SIM_LIST_FILE}")
    with open(SIM_LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Ensure we get the parent dir if list contains .../output/
                clean_path = os.path.normpath(line)
                if clean_path.endswith('output'):
                    clean_path = os.path.dirname(clean_path)
                run_paths.append(clean_path)

    if not run_paths:
        print(f"FATAL ERROR: No valid paths found in {SIM_LIST_FILE}"); return
    
    print(f"Found {len(run_paths)} simulation runs. Starting parallel processing with {NUM_PROCESSES} workers...")
    
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(obs_data,)) as pool:
        list(tqdm(pool.imap_unordered(process_single_run, run_paths), total=len(run_paths)))

    print("\n✅ All runs processed successfully!")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
