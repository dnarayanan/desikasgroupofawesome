#!/bin/bash
#SBATCH --job-name=10407_044
#SBATCH --mail-type=ALL
#SBATCH --mail-user=desika.narayanan@gmail.com
#SBATCH --time=96:00:00
####SBATCH --partition=hpg2-compute
#SBATCH --partition=hpg-default,hpg-turin,hpg-milan
#SBATCH --ntasks=512
####SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-socket=16
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=5gb
#SBATCH --account=narayanan
#SBATCH --qos=narayanan-b
#SBATCH --constraint=el9

module purge
module load ufrc
module load intel/2025.1.0
module load openmpi/5.0.7
module load hdf5/1.14.6
module load gsl/2.8
module load grackle/3.4.0
module load fftw/3.3.10
module load gsl



DATADIR=$SLURM_SUBMIT_DIR

export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="^vader,tcp,openib"
export OMPI_MCA_oob_tcp_listen_mode="listen_thread"

srun  --cpus-per-task=1 --mpi=${HPC_PMIX}   ./arepo/Arepo param.txt 1 1> output/OUTPUT  2> output/ERROR


