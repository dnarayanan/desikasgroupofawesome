Running Galaxy Simulations
**********
.. contents:: Section Contents
    :local:

Cosmological Simulations (Large-Box)
============

Initial Conditions With MUSIC
-----------------

The very first thing we'll need to do is to set up initial conditions
with MUSIC (https://www-n.oca.eu/ohahn/MUSIC/).  Please note: the
following instructions for MUSIC are for a non-zoom in cosmological
simulation only.  Please see below for initial conditions generation
for cosmological zoom in simulations.

Unfortunately, the IC generation is slightly different for Gizmo vs
Arepo because why would things be easy.  Where differences exist,
we'll note them.


For MUSIC, you'll
need a few libraries (compiler, GSL and FFTW loaded at the least).  I
suggest having your flags and paths set in the compiler as something
like::

  ##############################################################################
  ### compiler and path settings
  CC      = icc
  OPT     = -Wall -Wno-unknown-pragmas -O3 -g -mtune=native
  CFLAGS  =
  LFLAGS  = -lgsl -lgslcblas #-ldrfftw_threads
  CPATHS  = -I./src -I$(HOME)/local/include -I/opt/local/include -I/usr/local/include -I$(HPC_FFTW_INC) -I$(HPC_GSL_INC)
  LPATHS  = -L$(HOME)/local/lib -L/opt/local/lib -L/usr/local/lib -L$(HPC_FFTW_LIB) -L$(HPC_GSL_LIB)

So that the code automagically looks for whatever is added to your path when you module load the libraries.  In principle, you can just load modules before compiling like (as of 10/12/23, this is known to work for RHEL8 for music git hash 6747c54f3b73ec36719c265fd96362849a83cb45)::



  module load intel/2020.0.166  openmpi/4.1.5  hdf5/1.14.1 git/2.30.1  cmake/3.26.4 fftw/3.3.10  gsl/2.7 libz/1.2.11

**GIZMO:** The next thing you'll need is a configuration file for MUSIC.  Let's
set up a config file for a 25/h Mpc (side-length) and 512^3 (particle number) box.  The config file could look something like this::

  [desika.narayanan@login1 ICs]$ pwd
  /orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/ICs

  [desika.narayanan@login1 ICs]$ cat ics_m25n512.conf
  [setup]
  boxlength		= 25
  zstart			= 249
  levelmin		= 9
  levelmin_TF		= 9
  levelmax		= 9
  padding			= 8
  overlap			= 4
  ref_center              = 0,0,0
  ref_extent              = 0,0,0
  align_top		= no
  baryons			= yes
  use_2LPT		= no
  use_LLA			= no
  periodic_TF		= yes
  
  [cosmology]
  Omega_m			= 0.3
  Omega_L			= 0.7
  Omega_b			= 0.048
  H0			= 68.0
  sigma_8			= 0.82
  nspec			= 0.97
  transfer		= eisenstein
  
  [random]
  seed[9]			= 8675309
  
  [output]
  ##Gadget-2 (type=1: high-res particles, type=5: rest)
  format			= gadget2
  filename		= ics_m25n512
  gadget_usekpc		= yes
  gadget_usemsol		= no

  [poisson]
  fft_fine		= yes
  accuracy		= 1e-5
  pre_smooth		= 3
  post_smooth		= 3
  smoother		= gs
  laplace_order		= 6
  grad_order		= 6

Now note, there are a ton of options not listed here (that work both
with other hydrocodes than gadget-oids, as well as even for gadget
itself, and you should check out the MUSIC manual for those).  But in
short, the [setup] region of this tells you some obvious basics -- box
size, what redshift should the IC be set up for, what is the
coordinate system, etc.  The levelmin/max stuff is the particle count
-- so 9==2^9==512.  Similarly, we set that we want baryons (unless, of
course, we don't...) and our cosmology.  Important: this cosmology
will need to be the same as what we use in our actual hydro simulation.

**Arepo:** For Arepo things are slightly different.  Here, Arepo
 actually adds the baryons itself, so we have to generate a file with
 no baryons.  This looks different from the aforementioned Gizmo MUSIC
 IC in the following way::

   baryons			= no
   format			= arepo_double
   ##gadget_usekpc		= yes
   ##gadget_usemsol		= no

 note what is commented out above.

Once this config file is set, we need to actually run MUSIC on the config file to create the IC::

  [desika.narayanan@login1 ICs]$ pwd
  /orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/ICs

  [desika.narayanan@login1 ICs]$ cat music.job
  #!/bin/bash
  #SBATCH --job-name=music
  #SBATCH --output=music.o
  #SBATCH --error=music.e
  #SBATCH --mail-type=ALL
  #SBATCH --mail-user=desika.narayanan@gmail.com
  #SBATCH --time=36:00:00
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=32
  #SBATCH -N 1
  #SBATCH --mem-per-cpu=3800
  #SBATCH --account=narayanan
  #SBATCH --qos=narayanan-b
  
  module purge

  module load intel/2020.0.166  openmpi/4.1.5  hdf5/1.14.1 git/2.30.1  cmake/3.26.4 fftw/3.3.10  gsl/2.7 libz/1.2.11
  ./MUSIC ics_m25n512.conf

and the resultant file (which we set in the .conf file to be ics_m25n512) is the HDF5 initial condition for the simulation!

Compiling Gizmo
-----------------

We next want to run the actual gizmo simulation.  You'll need to clone
the gizmo repository.  Typically we've been using the SIMBA set of
galaxy physics, which you can find here:
https://bitbucket.org/romeeld/gizmo-mufasa/src/master/ (note, this is
private so you'll need access).

To comppile, the first thing we need is a Makefile that is set for our
system.  Edit Makefile.systype to have evverythign commented out
except the system we plan on using.  For example::

  # Select Target Computer
  #
  # Please copy this file to Makefile.systype and uncomment your
  # system. Don't commit changes to this file unless you add support for
  # a new system.
  #
  ###########
  #
  # This file was originally part of the GADGET3 code developed by
  #   Volker Springel (volker.springel@h-its.org).
  #
  #############
  
  ###################
  ## RT/RD SYSTEMS ##
  ###################
  #SYSTYPE="RTOSX"
  #SYSTYPE="ELGATO-GNU"
  #SYSTYPE="ELGATO-INTEL"
  #SYSTYPE="TIMON-PUMBAA_GNU"
  #SYSTYPE="TIMON-PUMBAA_OPEN64"
  #SYSTYPE="ursa"
  #SYSTYPE="ursa-open64"
  #SYSTYPE="fock"
  #SYSTYPE="fockgnu"
  SYSTYPE="hipergator-intel"
  #SYSTYPE="hipergator-gnu"
  #SYSTYPE="archer"
  #SYSTYPE="cosma-intel"
  #SYSTYPE="cosma-gnu"
  ################
  
  #SYSTYPE="Stampede"
  #SYSTYPE="Zwicky"
  #SYSTYPE="MacBookPro"
  #SYSTYPE="Quest"
  #SYSTYPE="odyssey"
  #SYSTYPE="SciNet"
  #SYSTYPE="Pleiades-Haswell"
  #SYSTYPE="Pleiades-SIBridge"
  #SYSTYPE="Ranger_intel"
  #SYSTYPE="Ranger_pgi"
  #SYSTYPE="Darwin"
  #SYSTYPE="Magny"
  #SYSTYPE="Magny-Intel"
  #SYSTYPE="OpenSuse"
  #SYSTYPE="OpenSuse64"
  #SYSTYPE="HLRB2"
  #SYSTYPE="MPA"
  #SYSTYPE="VIP"
  #SYSTYPE="Ubuntu"
  #SYSTYPE="MBM"
  #SYSTYPE="OpteronMPA-Gnu"
  #SYSTYPE="OpteronMPA-Intel"
  #SYSTYPE="Centos5-intel"
  #SYSTYPE="Kolob"
  #SYSTYPE="Centos5-Gnu"
  #SYSTYPE="OPA-Cluster64-Intel"


Where, here, we are obviously saying we'll compile using intel
compilers on HPG.  The next thing to do is to ensure that there are
actually system directives in the Makefile to actually compile!   For example, in the Makefile, have something like::

  ifeq ($(SYSTYPE),"hipergator-intel")
  CC   =  mpicc
  CXX  =  mpicxx
  FC   =  $(CC)
  OPT += -DH5_USE_16_API #-DCONFIG_BFLOAT_8
  #GSL_INCL    = -I$(HPC_GSL_INC)
  GSL_INCL    = -I/apps/intel/2018.1.163/gsl/2.4/include
  GSL_LIBS    = -L$(HPC_GSL_LIB)
  FFTW_HOME   = /apps/intel/2018.1.163/openmpi/3.1.0/fftw/2.1.5/
  FFTW_INCL   = -I$(FFTW_HOME)/include
  FFTW_LIBS   = -L$(FFTW_HOME)/lib64
  HDF5LIB     = -L$(HPC_HDF5_LIB) -lhdf5
  HDF5INCL    = -I$(HPC_HDF5_INC)
  BLAS_LIBS   = -L$(HPC_MKL_LIB) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
  GRACKLEINCL = -I$(HPC_GRACKLE_INC)
  GRACKLELIBS = -L$(HPC_GRACKLE_LIB) -lgrackle

Finally, we'll need to make some decisions about how to actually run
gizmo, given the physics that is implemented in the fork that we have.
This is really going to depend on your specific needs, so there's no
catch-all solution here. You can get the default Config.sh from the
simba-gizmo site.


Now, we should be able to compile!  Load the following modules, and
compile!::

  module purge
  module load intel/2018
  module load hdf5/1.10.1
  module load openmpi/3.1.2
  module load gsl/2.4
  module load fftw/2.1.5
  module load grackle

Once it's compiled, there is a parameter file to edit.  This will
point to your IC file, your output directory.  Some other things
you'll need to think about are the softening lengths: a reasonable
default is box length/particles per side/200 (in Mpc).  There's a nice
conversation in slack about this:
https://desikasgroupofawesome.slack.com/archives/C5HBZLSKX/p1643211197032300


These things are included in the param file, which will be seperate for each simulation. You can find an example param file for a simba simulation at::


  /orange/narayanan/s.lower/simba/m25n256_dm/zooms/track_dust_parms/run5_halo0_track_dust.param


We'll highlight some important parts of the param file::

  %---- ICs and Output
  InitCondFile /orange/narayanan/s.lower/simba/m25n256_dm/zooms/ICs/run5_halo0_ml10
  OutputDir /blue/narayanan/s.lower/zoom_temp/run5_halo0/

  %---- File formats
  ICFormat  1 % 1=binary, 3=hdf5, 4=cluster
  SnapFormat 3 % 1=binary, 3=hdf5

  %---- Output parameters
  RestartFile         restart
  SnapshotFileBase      snapshot
  OutputListOn        1 % =1 to use list in OutputListFilename
  OutputListFilename       /orange/narayanan/s.lower/simba/m25n256_dm/zooms/15Myr_cadence_z1p5.txt
  NumFilesPerSnapshot     1
  NumFilesWrittenInParallel  1 % must be < N_processors & power of 2


The top part will point to your IC file and where you want to save the simulation snapshots. The second part specifies the format that the ICs are in and the format you want the snapshots to be in. The last part has a very important parameter called `OutputListFilename`, which sets the times (in simulation scale factor units) when the snapshots will be written. This is a pretty important scheme to choose carefully, since the science you want to do with these simulations could heavily depend on the time resolution of the snapshots (i.e., if you want to track the accretion of gas onto early halos, you'll want fine time resolution in the first billion years). The file in this example tells Gizmo to write snapshots every 15 Myr starting at z=20. If the simulation runs to z=1.5, this will output ~200 snapshots. So your choice of time resolution is also dependent on storage requirements. So keep this in mind!::

  %---- Cosmological parameters
  ComovingIntegrationOn  1    % is it cosmological? (yes=1, no=0)
  BoxSize         25000. % in code units
  Omega0         0.3  % =0 for non-cosmological
  OmegaLambda       0.7  % =0 for non-cosmological
  OmegaBaryon       0.048  % =0 for non-cosmological
  HubbleParam       0.68   % little 'h'; =1 for non-cosmological runs
  %---- Accuracy of time integration
  MaxSizeTimestep     0.005  % in code units, set for your problem
  MinSizeTimestep     1.0e-12 % set this very low, or risk stability
  %---- Tree algorithm, force accuracy, domain update frequency
  TreeDomainUpdateFrequency  0.005    % 0.0005-0.05, dept on core+particle number


  %---- Gravitational softening lengths
  %----- Softening lengths per particle type. If ADAPTIVE_GRAVSOFT is set, these
  %-------- are the minimum softening allowed for each type -------
  %-------- (units are co-moving for cosmological integrations)
  SofteningGas  0.15  % gas (type=0) (in units above, =1 pc softening)
  SofteningHalo  0.15  % dark matter/collisionless particles (type=1)
  SofteningDisk  0.15  % collisionless particles (type=2)
  SofteningBulge 0.15  % collisionless particles (type=3)
  SofteningStars 0.15  % stars spawned from gas (type=4)
  SofteningBndry 0.15  % black holes (if active), or collisionless (type=5)
  %---- if these are set in cosmo runs, SofteningX switches from comoving to physical
  %------- units when the comoving value exceeds the choice here
  SofteningGasMaxPhys   0.15  % switch to 0.5pc physical below z=1
  SofteningHaloMaxPhys  0.15
  SofteningDiskMaxPhys  0.15
  SofteningBulgeMaxPhys  0.15
  SofteningStarsMaxPhys  0.15
  SofteningBndryMaxPhys  0.15
  %----- parameters for adaptive gravitational softening
  AGS_DesNumNgb      64 % neighbor number for calculating adaptive gravsoft

  
Next, we need to match the boxsize and cosmology to the values used to generate the ICs. And finally, you need to adjust the softening lengths to the box size / resolution as mentioned above.


Then, you should be in business to run!  This is an
example from one of Sidney's zooms::

  [desika.narayanan@login1 zooms]$ pwd
  /orange/narayanan/s.lower/simba/m25n256_dm/zooms
  [desika.narayanan@login1 zooms]$ more simba_ompi.job
  #!/bin/bash
  #SBATCH --job-name=r31_ml11
  #SBATCH --output=run_logs/run31_ml11.log
  #SBATCH --mem-per-cpu=3900
  #SBATCH --time=96:00:00
  #SBATCH --mail-user=s.lower@ufl.edu
  #SBATCH --mail-type=ALL
  #SBATCH --ntasks=512
  #SBATCH --ntasks-per-socket=8
  #SBATCH --distribution=cyclic:cyclic
  #SBATCH --cpus-per-task=1
  ##SBATCH --partition=hpg-default
  #SBATCH --account=narayanan
  #SBATCH --qos=narayanan-b
  ##SBATCH --account=astronomy-dept
  ##SBATCH --qos=astronomy-dept-b
  
  
  module purge
  module load intel/2018
  module load hdf5/1.10.1
  module load openmpi/3.1.2
  module load gsl/2.4
  module load fftw/2.1.5
  module load grackle
  
  export OMPI_MCA_pml="ucx"
  export OMPI_MCA_btl="^vader,tcp,openib"
  export OMPI_MCA_oob_tcp_listen_mode="listen_thread"
  
  DATADIR=$SLURM_SUBMIT_DIR
  cd $DATADIR/gizmo_simba_track_dust
  srun --mpi=pmix_v2  GIZMO $DATADIR/ml11_zoom_param_files/run31_halo0_ml11.param
  

Compiling Arepo
-----------------
Arepo is similar to Gizmo with the following updates.  All notes are as of June 2025 on HiPerGator:

First, HiPerGator2 [note, this only works for RedHat EL8, which will be deprecated as of August 2025]::

  module purge
  module load intel/2020.0.166
  module load openmpi/4.1.6
  module load python/3.11.4
  module load fftw/3.3.10
  module load hdf5/1.14.5
  module load grackle/3.2.1
  module load gsl/2.6

Second, HiPerGator3 [RedHat EL8]::
  module purge
  module load ufrc
  module load intel/2020.0.166
  module load openmpi/4.1.6
  module load python/3.11.4
  module load fftw/3.3.10
  module load hdf5/1.14.1
  module load grackle/3.2.1
  module load gsl/2.6


Finally, HiPerGator3 [RedHat EL9]::
  module purge
  module load ufrc
  module load intel/2025.1.0
  module load openmpi/5.0.7
  module load hdf5/1.14.6
  module load gsl/2.8
  module load grackle/3.4.0
  module load fftw/3.3.10
  module load gsl
  
  
Like with Gizmo you'll need to look at someone else's Config.sh to
compile to set the correct physics. This said there is one important
note: to use the MUSIC ICs as described above (with baryons off),
we'll need this set for sure in the Config.sh::
  
  GENERATE_GAS_IN_ICS

Finally to compile, we type::
  
  make clean
  make build




Cosmological Simulations (Zoom-in)
============

Running a cosmological zoom-in simulation is more or less the same as
a large box simulation, though with one major difference: the IC file
created by music is rather different.  As a summary: For a zoom-in
simulation, we want to have first run a large box low-resolution dark
matter only simulation.  From that large box simulation, we then
identify a halo with Caesar that we want to "zoom-in" on.  With
Caesar, we will create a 'mask' around this halo which identifies
region we want to re-simulate at high resolution.  This information is
then fed into MUSIC which will split the particles that are in this
high resolution mask N times (in order to obtain a desired particle
resolution), and everything outside of this mask (from the parent DM
only large box simulation) will remain at low-resolution.  This allows
us to capture large scale torques/gravitational effects on the zoom
galaxy of interest, while maintaining high particle resolution within
the zoomed-in halo.  

To write the mask, we will use CAESAR in the following manner::

  import numpy as np
  import caesar,yt
  
  #modeled after /orange/narayanan/s.lower/simba/m25n256_dm/zooms/halo_masks/write_halo_mask.py
  
  snapshot = '/orange/narayanan/s.lower/simba/m25n256_dm/output/run1/snapshot_008.hdf5'
  icfile = '/orange/narayanan/s.lower/simba/m25n256_dm/IC_stuff/run1_ICs/ics_m25n256_Run1.0'
  caesarfile = '/orange/narayanan/s.lower/simba/m25n256_dm/output/run1/Groups/caesar_snapshot_008.hdf5'
  halonum =0
  
  outfile = 'run1_halo0.mask.txt'
  

  obj = caesar.load(caesarfile)
  ic = icfile
  ds = yt.load(snapshot)
  ic_ds = yt.load(ic)
  obj.yt_dataset = ds
  obj.halos[halonum].write_IC_mask(ic_ds,outfile,radius_type='total_half_mass')

  
Where icfile is the initial conditions MUSIC file from the parent dark
matter only simulation, and the snapshot is the snapshot we're
building the zoom from.  This snapshot should represent the latest
possible redshift you are interested in running the zoom to (since if
you run past this, then low-res particles will eventually fall into
the halo and contaminate it).  There's an art to choosing this final
redshift: you obviously don't want to short change yourself and pick a
final redshift that's too large, only to wish you could run your zoom
further.  At the same time, the lower the redshift of this final
snapshot (that we select the halo to resimulate from), the more
particles there will be in it, and the harder the zoom in simulation
will be to run.

There is additionally some art to choosing the radius_type above.  The
larger the radius_type, the less likely we are to suffer contamination
down the line.  At the same time, too large of a radius will not only
slow the simulation down, but we can't have a radius larger than the
box size or else we'll get the following error from MUSIC::

   - ERROR: On level 9, subgrid is larger than half the box. This is not allowed!
     terminate called after throwing an instance of 'std::runtime_error'
     what():  Fatal: Subgrid larger than half boxin zoom.
     Aborted (core dumped)

The options for radii types for dark halos are printed below (though note the baryon ones won't work if your initial low-res simulation is DM only)::
  dict_keys(['baryon_half_mass', 'baryon_r20', 'baryon_r80', 'dm_half_mass', 'dm_r20', 'dm_r80', 'total_half_mass', 'total_r20', 'total_r80'])

[more to fill in yet - just a place holder for now]



Idealized Galaxy Simulations
============

First, all of the relevant files live on a private GitHub space at
https://github.com/dnarayanan/arepo_ics - be sure to ask if you don't
have permissions here.

Initial Conditions
-----------------

Make New Disk:

The first thing we'll need to do is run MakeNewDisk in order to make
an idealized galaxy disk IC.  You can find some examples on HiPerGator
here::

  /home/paul.torrey/InitialConditions/PhilsSpecificICs/MW
  /home/paul.torrey/InitialConditions/PhilsSpecificICs/SMC
  /home/paul.torrey/InitialConditions/PhilsSpecificICs/Sbc
  /home/paul.torrey/InitialConditions/PhilsSpecificICs/HiZ
  
(though note because of the size only the MW example is on GitHub).  You have to edit the resolution in main.c and then::

  make clean
  make

This is light weight code and can usually run from even a login node.  You can just run from something like::

  ./MakeHubbleType ./output/MW.dat

which produces your basic initial condition file.

**Adding Backgrounds for Arepo Simulations** If we were running a
gizmo simulation, we could stop right here.  Note, however, if we are
running an arepo simulation, this will require a background grid to be
added to the IC.  To do this, we use sbatch_makeIC.sh which is a job
script that calls arepo from the arepo_addbg directory.  Note, you'll
need a working executable for arepo in that directory.  There are
param files that are necessary for adding the background -- these
parameter files will have the params you'll use at simulation runtime,
and there are examples in::

  /blue/narayanan/desika.narayanan/MakeGalaxy/arepo_addbg

as well as the GitHub repo.  An example of this .sh file (for posterity) is::

  (pd4env_gcc) [desika.narayanan@login2 MakeGalaxy]$ more sbatch_makeIC.sh
  #!/bin/bash
  #SBATCH --job-name=makeIC
  #SBATCH --mail-type=ALL
  #SBATCH --mail-user=desika.narayanan@gmail.com
  #SBATCH --time=1:00:00
  #SBATCH --nodes=8
  #SBATCH --tasks-per-node=16
  #SBATCH --ntasks-per-socket=16
  #SBATCH --cpus-per-task=1
  #SBATCH --distribution=cyclic:cyclic
  #SBATCH --mem-per-cpu=8gb

  #SBATCH --partition=hpg-default
  #SBATCH --account=narayanan
  #SBATCH --qos=narayanan


  module purge
  module load intel/2020.0.166
  module load gsl/2.6
  module load openmpi/4.1.6
  module load hdf5/1.14.1

  DATADIR=$SLURM_SUBMIT_DIR
  
  export OMPI_MCA_pml="ucx"
  export OMPI_MCA_btl="^vader,tcp,openib"
  export OMPI_MCA_oob_tcp_listen_mode="listen_thread"
  
  srun     ./arepo_addbg/Arepo   arepo_addbg/param_MW_ultra_lowres.txt 0        1> output_makeIC/OUTPUT  2> output_makeIC/ERROR



When you run this background addition, it
will automatically make a new file that has the appendate
"--with-grid.hdf5".  For example if your input file was called
"MW_lr.dat", your output file (assuming output type 3 is used) will be
called "MW_lr.dat-with-grid.hdf5"

**Further Modifying ICs for Dust**

We have one final step which is to move all type 2 and type 3 particles to type 4.  We do this with a script written by Qi Li called modifyIC.py -- there's an example on GitHub as well as at::

  /blue/narayanan/desika.narayanan/MakeGalaxy

Run this and we should have a new IC that is ready for Arepo as well!

Run Time
-----------------

You can find examples of the parameter files in the GitHub subdirectory idealized_repo_example or at::

  /blue/narayanan/desika.narayanan/arepo_runs/idealized/MW_ultra_lowres
