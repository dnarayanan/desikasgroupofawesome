#!/bin/bash					
#SBATCH -J music_ICs
#SBATCH --mem-per-cpu=250000
#SBATCH --time=96:00:00
#SBATCH --mail-user=desika.narayanan@gmail.com
#SBATCH --mail-type=START,FAIL,END
#SBATCH --partition=hpg-default
#SBATCH --ntasks=1
#SBATCH --account=paul.torrey
#SBATCH --qos=paul.torrey

module purge
module load intel/2025.1.0  openmpi/5.0.7  hdf5 fftw gsl

#hdf5/1.14.1 fftw/3.3.10  gsl/2.8


export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="^vader,tcp,openib"
export OMPI_MCA_oob_tcp_listen_mode="listen_thread"

####a/home/paul.torrey/Projects/ISM_Boxes/ICs/ohahn-music-afefabeea948/MUSIC /blue/narayanan/paul.torrey/DustICs/ics_m25n128.conf

#./music/MUSIC halo_configs/ics_run0_halo1.ml12.conf
#./music/MUSIC ics_run0_halo500.ml13.conf >  halo500.ml13.out
#./music/MUSIC ics_run0_halo500.ml12.conf > halo500.ml12.out
#./music/MUSIC ics_run0_halo750.ml12.conf > halo750.ml12.out
./music/MUSIC ics_run0_halo300.ml12.conf > halo300.ml12.out

