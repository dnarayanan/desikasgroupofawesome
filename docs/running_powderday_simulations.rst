Running POWDERDAY Simulations
**********
.. contents:: Section Contents
    :local:
    
Is POWDERDAY Working?
============

First, if you have not already, you'll need to verify that your POWDERDAY is working properly.
To do so, follow the checks at the POWDERDAY installation docs (https://powderday.readthedocs.io/en/latest/).
My usual test is the SKIRT GIZMO Milky Way zoom simulation. Briefly, you'll want to follow these commands after loading the appropriate modules::

	cd powderday
	python pd_front_end.py tests/SKIRT/gizmo_mw_zoom parameters_master_gizmo parameters_model_gizmo

If all goes well, this should finish without a problem. You should also check the output SED following the POWDERDAY installation docs. 
The 'parameters_master' file you use when running POWDERDAY will dictate exactly what important parameters you are giving to POWDERDAY (e.g. whether to have AGN on, IMF being used, etc.).
The 'parameters_model' file includes information such as where the results will be stored, and where to find the galaxy in the simulation.
If you're unsure about whether everything in these two files is good, post in the Slack.


Workflow Setup
============
Retreive some version of Sidney's scripts for automating (https://github.com/smlower/sl_simulation_tools) the POWDERDAY runs. 
Dhruv's modified versions can be found in ``/home/d.zimmerman/sl_simulation_tools/`` (and hopefully on GitHub soon). These are already set up to help
automate everything and will make your life significantly easier. 

Initial Setup Work
============


Filtering Galaxies [Optional]
-----------------

At this point, you hopefully have a CAESAR file to reference for
galaxies for the appropriate snapshots.  If not, refer to the CAESAR
docs (https://caesar.readthedocs.io/en/latest/) for how to get that
done.  At this point, you may wish to filter galaxies out of your
cosmological simulation.  This technique, originally developed by Ben
Kimock and Sidney Lower, allows you to just grab the gas and star
particles out of the parent snapshot, and create a mini snapshot with
just an individual galaxy in it.  (Note: this process does not
automatically include PartType3 dust or PartType5 black holes, though
it should be reasonably straight forward to update this as needed).

The value in filtering a snapshot is that it ensures that all of the
emission from powderday *only* comes from particles associated with
the galaxy in CAESAR.  This helps when (e.g.) comparing physical
properties as they would be derived from the observations with the
true physical properties.  A script that will filter all of the
galaxies in an individual CAESAR snapshot as modified by Dhruv Zimmerman is below::

  import h5py
  import caesar
  import sys
  import glob
  import numpy as np
  import tqdm
  import os
  
  ###########
  # Line arguments
  ###########
  snapshot_path = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/snapshot_'
  snap_num = 59
  output_path = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/filtered_snaps/snap'+str(snap_num).zfill(3)
  caesar_file = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/Groups/caesar_0059_z7.490.hdf5'
  
  #see if the output path exists, and if not, make it

  if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("creating output directory: "+output_path)
  
	
  obj = caesar.load(caesar_file)
  snap_str = str(snap_num).zfill(3)
  
  input_file = h5py.File(snapshot_path+str(snap_str)+'.hdf5', 'r')
  

  galcount = len(obj.galaxies)
  for galaxy in range(galcount):
        print()
        print("GALAXY NUM:",str(galaxy))
        print()
        glist = obj.galaxies[int(galaxy)].glist
        slist = obj.galaxies[int(galaxy)].slist


        with h5py.File(output_path+'galaxy_'+str(galaxy)+'.hdf5', 'w') as output_file:
            output_file.copy(input_file['Header'], 'Header')
            print('starting with gas attributes now')
            output_file.create_group('PartType0')
            for k in tqdm.tqdm(input_file['PartType0']):
                output_file['PartType0'][k] = input_file['PartType0'][k][:][glist]
            print('moving to star attributes now')
            output_file.create_group('PartType4')
            for k in tqdm.tqdm(input_file['PartType4']):
                output_file['PartType4'][k] = input_file['PartType4'][k][:][slist]


        print('done copying attributes, going to edit header now')
        outfile_reload = output_path+'galaxy_'+str(galaxy)+'.hdf5'

        re_out = h5py.File(outfile_reload,'r+')
        re_out['Header'].attrs.modify('NumPart_ThisFile', np.array([len(glist), 0, 0, 0, len(slist), 0]))
        re_out['Header'].attrs.modify('NumPart_Total', np.array([len(glist), 0, 0, 0, len(slist), 0]))

        re_out.close()




Galaxy Positions
-----------------

The next, relatively minor, part of the setup process requires running the ``galaxy_positions.py`` script. The purpose of this script is to use the
newly generated filtered snapshots and simply generate a list of the positions of the center of the galaxies.
Again, Dhruv's current version looks like this::

	import h5py
	import numpy as np
	import sys, os
	import numpy as np
	import glob
	import tqdm

	##############
	# Line arguments
	###############
	snap = int(sys.argv[1])
	snap_dir = '/orange/narayanan/[...]' #where are the filtered galaxies?
	outfile = '/orange/narayanan/[...]'+'_gal_positions.npz' #where do you want the output to go?
	################

	pos = {}
	ngalaxies = {}
	infiles = sorted(glob.glob(snap_dir+'/galaxy_*.hdf5'))
	count = 0
	for i in tqdm.tqdm(range(len(infiles))):
	    try:
	        infile = h5py.File(snap_dir+'/galaxy_'+str(i)+'.hdf5', 'r')
	    except:
	        print(str(i))
	        continue
	    count+=1
	    pos['galaxy'+str(i)] = {}


	    gas_masses = infile['PartType0']['Masses']
	    gas_coords = infile['PartType0']['Coordinates']
	    star_masses = infile['PartType4']['Masses']
	    star_coords = infile['PartType4']['Coordinates']
	    total_mass = np.sum(gas_masses) + np.sum(star_masses)

	    x_pos = (np.sum(gas_masses * gas_coords[:,0]) + np.sum(star_masses * star_coords[:,0])) / total_mass
	    y_pos = (np.sum(gas_masses * gas_coords[:,1]) + np.sum(star_masses * star_coords[:,1])) / total_mass
	    z_pos = (np.sum(gas_masses * gas_coords[:,2]) + np.sum(star_masses * star_coords[:,2])) / total_mass
	
	
	    pos['galaxy'+str(i)]['snap'+str(snap)] = np.array([x_pos, y_pos, z_pos])
	    infile.close()
	ngalaxies['snap'+str(snap)] = count


	print("SAVING")
	np.savez(outfile, ngalaxies=ngalaxies, pos=pos)




Setting up POWDERDAY to Run on Filtered Galaxies
-----------------
At this point, hopefully you have successfully filtered the galaxies in your CAESAR file into individual galaxy files and one file storing the center locations of these galaxies. Now you're all set to worry about POWDERDAY. You are currently missing some parameters_model scripts for your POWDERDAY run. To resolve this, there are two important files you'll need to use from https://github.com/smlower/sl_simulation_tools (only one directly) to get all set up: ``powderday_setup.py`` and ``cosmology_setup_all_cluster.hipergator.sh``. The python script will call the bash script with location and temperature information pulled from the simulation. The bash script will automatically generate the parameters model files for you for each galaxy with the given information at the given locations. Dhruv’s current versions of these scripts are as follows::

	# powderday_setup.py
	#purpose: to set up slurm files and model *.py files from the
	#positions written by caesar_cosmology_npzgen.py for a cosmological
	#simulation.  This is written for the University of Florida's
	#HiPerGator2 cluster.
	import numpy as np
	from subprocess import call
	import sys
	
	nnodes=1
	snap_dict = {'74':6.014,'80':5.530,'87':5.024,'95':4.515,'104':4.015,'115':3.489,'127':3.003,'142':2.496,'160':2.0,'183':1.497,'212':1.007,'252':0.501,'305':0.0} # edit this list as you see fit for the snapshots you use
	simb_run = "m25n512" # what SIMBA box are you using?
	snap_num = sys.argv[1] # takes the snapshot as an in-line parameter – important for the bash scripts
	snap_redshift = snap_dict[snap_num]
	npzfile = '/orange/narayanan/[…]/snap'+str(snap_num)+'_gal_positions.npz' # where did you put the galaxy positions file?
	model_dir_base = '/orange/narayanan/[…]' # where do you want your POWDERDAY parameters model files to go?
	out_dir_base = '/orange/narayanan/[…]’ # where do you want your SED files to go when POWDERDAY is finished?
	hydro_dir = '/orange/narayanan/[…]' # where are your filtered galaxies?
	hydro_dir_remote = hydro_dir
	model_run_name='simba_m25n512' # shorthand for what you are running
	#################
	COSMOFLAG=0 #flag for setting if the gadget snapshots are broken up into multiples or not and follow a nomenclature snapshot_000.0.hdf5
	FILTERFLAG = 1 #flag for setting if the gadget snapshots are filtered or not, and follow a nomenclature galaxy_1800.hdf5 – this can easily be changed if you prefer some other naming convention
	SPHGR_COORDINATE_REWRITE = True
	#===============================================
	if (COSMOFLAG == 1) and (FILTERFLAG == 1):
    		raise ValueError("COSMOFLAG AND FILTER FLAG CAN'T BOTH BE SET")
	data = np.load(npzfile,allow_pickle=True)
	pos = data['pos'][()] #positions dictionary
	#ngalaxies is the dict that says how many galaxies each snapshot has, in case it's less than NGALAXIES_MAX
	ngalaxies = data['ngalaxies'][()]

	for snap in [snap_num]: # artifact of old code, does not have to be a loop
		model_dir = model_dir_base
		model_dir_remote = model_dir
		tcmb = 2.73*(1.+snap_redshift) # will be important at higher z
		NGALAXIES = ngalaxies['snap'+str(snap)]
		
		for nh in range(NGALAXIES):
			try:
				xpos = pos['galaxy'+str(nh)]['snap'+str(snap)][0] # extra positional information
			except: continue
			
			ypos = pos['galaxy'+str(nh)]['snap'+str(snap)][1]
			zpos = pos['galaxy'+str(nh)]['snap'+str(snap)][2]
			#print("CALLING")
			cmd = "./cosmology_setup_all_cluster.hipergator.sh "+str(nnodes)+' '+model_dir+' '+hydro_dir+' '+out_dir_base+' '+model_run_name+' '+str(COSMOFLAG)+' '+str(FILTERFLAG)+' '+model_dir_remote+' '+hydro_dir_remote+' '+str(xpos)+' '+str(ypos)+' '+str(zpos)+' '+str(nh)+' '+str(snap)+' '+str(tcmb)
			call(cmd,shell=True) # call the bash script with the calculated numbers as parameters
        		#print("CALLED")

	# start of bash script

	#!/bin/bash

	#Powderday cluster setup convenience script for SLURM queue manager
	#on HiPerGator at the University of FLorida.  This sets up the model
	#files for a cosmological simulation where we want to model many
	#galaxies at once.

	#Notes of interest:

	#1. This does *not* set up the parameters_master.py file: it is
	#assumed that you will *very carefully* set this up yourself.

	#2. This requires bash versions >= 3.0.  To check, type at the shell
	#prompt:

	#> echo $BASH_VERSION
	# grab the numbers
	n_nodes=$1
	model_dir=$2
	hydro_dir=$3
	out_dir=$4
	model_run_name=$5
	COSMOFLAG=$6
	FILTERFLAG=$7
	model_dir_remote=$8
	hydro_dir_remote=$9
	xpos=${10}
	ypos=${11}
	zpos=${12}
	galaxy=${13}
	snap=${14}
	tcmb=${15}

	echo "processing model file for galaxy,snapshot:  $galaxy,$snap"
	
	#clear the pyc files
	rm -f *.pyc

	#set up the model_**.py file
	echo "setting up the output directory in case it doesnt already exist"
	echo "snap is: $snap"
	echo "model dir is: $model_dir"
	mkdir $model_dir
	
	filem="$model_dir/snap${snap}_galaxy${galaxy}.py"
	echo "writing to $filem"
	rm -f $filem
	
	# setting up header
	echo "#Snapshot Parameters" >> $filem
	echo "#<Parameter File Auto-Generated by setup_all_cluster.sh>" >> $filem
	echo "snapshot_num =  $snap" >> $filem 
	echo "galaxy_num = $galaxy" >>$filem
	echo -e "\n" >> $filem

	echo -e "galaxy_num_str = str(galaxy_num)" >> $filem

	# may need to include depending on how you converted to naming conventions
	#echo "if galaxy_num < 10:" >> $filem
	#echo -e "\t galaxy_num_str = '00'+str(galaxy_num)" >> $filem
	#echo -e "elif galaxy_num >= 10 and galaxy_num <100:" >> $filem
	#echo -e "\t galaxy_num_str = '0'+str(galaxy_num)" >> $filem
	#echo -e "else:" >> $filem
	#echo -e "\t galaxy_num_str = str(galaxy_num)" >> $filem
	
	echo -e "\n" >>$filem

	echo -e "snapnum_str = str(snapshot_num)" >> $filem

	echo -e "\n" >>$filem
	if [ $COSMOFLAG -eq 1 ]
	then
    		echo "hydro_dir = '$hydro_dir_remote/snapdir_'+snapnum_str+'/'">>$filem
    		echo "snapshot_name = 'snapshot_'+snapnum_str+'.0.hdf5'" >>$filem
	elif [ $FILTERFLAG -eq 1 ] # you’ll be using this 99.9% of the time
	then
    		echo "hydro_dir = '$hydro_dir_remote/'">>$filem
    		echo "snapshot_name = 'galaxy_'+str(galaxy_num)+'.hdf5'">>$filem # change this line for filtered naming conventions
	else
    		echo "hydro_dir = '$hydro_dir_remote/'">>$filem
    		echo "snapshot_name = 'snapshot_'+snapnum_str+'.hdf5'" >>$filem
	fi


	echo -e "\n" >>$filem

	echo "#where the files should go" >>$filem
	echo "PD_output_dir = '${out_dir}/' ">>$filem # again, where you want things to go
	echo "Auto_TF_file = 'snap'+snapnum_str+'.logical' ">>$filem # COME BACK
	echo "Auto_dustdens_file = 'snap'+snapnum_str+'.dustdens' ">>$filem # COME BACK

	echo -e "\n\n" >>$filem 
	echo "#===============================================" >>$filem
	echo "#FILE I/O" >>$filem
	echo "#===============================================" >>$filem
	echo "inputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtin'" >>$filem
	echo "outputfile = PD_output_dir+'snap'+snapnum_str+'.galaxy'+galaxy_num_str+'.rtout'" >>$filem
	echo -e "\n\n" >>$filem
	echo "#===============================================" >>$filem
	echo "#GRID POSITIONS" >>$filem
	echo "#===============================================" >>$filem
	echo "x_cent = ${xpos}" >>$filem
	echo "y_cent = ${ypos}" >>$filem
	echo "z_cent = ${zpos}" >>$filem

	echo -e "\n\n" >>$filem
	echo "#===============================================" >>$filem
	echo "#CMB INFORMATION" >>$filem
	echo "#===============================================" >>$filem
	echo "TCMB = ${tcmb}" >>$filem
	# from here we make the job script that you can use
	echo "writing slurm submission master script file"
	qsubfile="$model_dir/master.snap${snap}.job"
	rm -f $qsubfile
	echo $qsubfile
	echo "#! /bin/bash" >>$qsubfile
	echo "#SBATCH --job-name=${model_run_name}.snap${snap}" >>$qsubfile
	echo "#SBATCH --output=pd.master.snap${snap}.o" >>$qsubfile
	echo "#SBATCH --error=pd.master.snap${snap}.e" >>$qsubfile
	echo "#SBATCH --mail-type=ALL" >>$qsubfile
	echo "#SBATCH --mail-user=[…]@ufl.edu" >>$qsubfile # your email
	echo "#SBATCH --time=48:00:00" >>$qsubfile
	echo "#SBATCH --tasks-per-node=32">>$qsubfile
	echo "#SBATCH --nodes=$n_nodes">>$qsubfile
	echo "#SBATCH --mem-per-cpu=3800">>$qsubfile
	echo "#SBATCH --account=narayanan">>$qsubfile
	echo "#SBATCH --qos=narayanan-b">>$qsubfile
	echo "#SBATCH --array=0-99">>$qsubfile # preferably modify with a % ‘max number of jobs’ when actually running, job # will correspond to galaxy number in some way
	echo -e "\n">>$qsubfile
	echo -e "\n" >>$qsubfile

	# the meat of the job script that actually tells SLURM what to do
	# get your modules loaded (make sure to modify with your own appropriate ones)
	echo "cd /home/d.zimmerman">>$qsubfile
	echo "module purge">>$qsubfile
	echo "source .bashrc">>$qsubfile
	echo "source activate master_env">>$qsubfile
	echo -e "\n">>$qsubfile
	echo "module load git">>$qsubfile
	#echo "module load gcc/12.2.0">>$qsubfile
	echo "module load intel/2020.0.166">>$qsubfile
	echo "module load openmpi/4.1.5">>$qsubfile
	echo "module load hdf5/1.14.1">>$qsubfile
	echo -e "\n">>$qsubfile

	echo "ID=\$(awk '{if(NR==(n+1)) print int(\$0)}' n=\${SLURM_ARRAY_TASK_ID} /orange/narayanan/d.zimmerman/simba/m25n512/snap${snap}/snap${snap}_gas_gals.txt)">>$qsubfile # Something Dhruv has used to only run POWDERDAY on galaxies with gas (you will need to set up the txt file if you want this), important if you are running over many galaxies in a simulation, if not, substitute subsequent ‘ID’ instances with ‘SLURM_ARRAY_TASK_ID’, which is a SLURM variable
	echo -e "\n">>$qsubfile
	# calling POWDERDAY
	echo "cd /home/d.zimmerman/powderday/">>$qsubfile
	echo "pd_front_end.py $model_dir_remote parameters_master_catalog snap${snap}_galaxy\${ID} > $out_dir/outlogs/snap${snap}_galaxy\${ID}.log">>$qsubfile
	echo "date"

Doing it All the Setup in One Go
-----------------
Dhruv’s modified scripts are constructed and intended so that one can run them for a bunch of snapshots at once given the CAESAR files and intended destinations. If you’re confident that you have the above scripts working all correctly, you can modify the bash scripts below to do everything you want in one go for all snapshots you care about. I would personally recommend filtering separately and then running the POWDERDAY setup as below as filtering will be the majority of the time usage and has different memory requirements, but it should not be a problem to run both as long as you adjust the job parameters appropriately. Note that if you want to use the m100 box, you should also be careful with both memory and time allocations.::

	# start of filter bash script


	#!/bin/bash
	#SBATCH --job-name=simba_filter_array
	#SBATCH --output=output.log
	#SBATCH --mail-type=ALL
	#SBATCH --mail-user=[…]@ufl.edu
	#SBATCH --ntasks=4
	#SBATCH --nodes=1
	#SBATCH --mem=60gb
	#SBATCH --account=narayanan
	#SBATCH --qos=narayanan
	#SBATCH --time=20:00:00
	#SBATCH --array=[…]


	date;hostname;pwd;
	cd /home/d.zimmerman
	module purge
	source .bashrc

	source activate master_env

	module load git
	#module load gcc/12.2.0
	module load intel/2020.0.166
	module load openmpi/4.1.5
	module load hdf5/1.14.1

	python /home/d.zimmerman/sl_simulation_tools-main/filter_simba_all.py $SLURM_ARRAY_TASK_ID

	date


	# start of powderday setup bash script

	#!/bin/bash
	#SBATCH --job-name=simba_pd_setup
	#SBATCH --output=output_pd_setup.log
	#SBATCH --mail-type=ALL
	#SBATCH --mail-user=[...]@ufl.edu
	#SBATCH --ntasks=4
	#SBATCH --nodes=1
	#SBATCH --mem=10gb
	#SBATCH --account=narayanan
	#SBATCH --qos=narayanan
	#SBATCH --time=20:00:00
	#SBATCH --array=87
	

	#74,104,127,160,212,305 - list of snapshots that correspond to array jobs


	date;hostname;pwd;
	cd /home/d.zimmerman
	module purge
	source .bashrc
	
	source activate master_env

	module load git
	#module load gcc/12.2.0
	module load intel/2020.0.166
	module load openmpi/4.1.5
	module load hdf5/1.14.1

	cd /home/d.zimmerman/sl_simulation_tools-main/

	python /home/d.zimmerman/sl_simulation_tools-main/galaxy_positions.py $SLURM_ARRAY_TASK_ID
	#python /home/d.zimmerman/caesar_good_gal_script.py $SLURM_ARRAY_TASK_ID
	python /home/d.zimmerman/sl_simulation_tools-main/powderday_setup.py $SLURM_ARRAY_TASK_ID
	date


The script for filtering galaxies for those with only gas is relatively simple and can be found below or at ``/home/d.zimmerman/caesar_good_gals_script.py``::

	import yt
	import caesar
	import numpy as np
	import sys
	import matplotlib.pyplot as plt
	
	simb_run = "m100n1024" # again, which SIMBA box you care aboute

	fileroot = '/orange/narayanan/d.zimmerman/simba/'+simb_run+'/caesar_cats/caesar_simba_' # where are your CAESAR files?
	saveroot = '/orange/narayanan/d.zimmerman/simba/'+simb_run+'/snap' # where do you want this to do?
	fileex='.hdf5'

	snapnums=[127,142,160,183,212,252,305] # list of snapshots
	num = int(sys.argv[1]) 
	
	#for num in snapnums: # you’ll want to comment out above and uncomment this to run this outside the above script setup
	caes_obj = caesar.load(fileroot+str(num)+fileex)
	gal_gasses = np.array([caes_obj.galaxies[i].masses['gas'] for i in range(len(caes_obj.galaxies))])
	gal_index_list = np.array(range(len(gal_gasses)),dtype=int)
	print(gal_index_list)
	good_gals = gal_index_list[gal_gasses > 0]
	print(good_gals)
	test_file = open(saveroot+str(num)+"/snap"+str(num)+"_gas_gals.txt","w")
	#for j in good_gals:
	np.savetxt(test_file,good_gals,fmt='%s') # save info into text file
	test_file.close()


With that, you simply need to copy over a ``parameters_master`` file to your directories containing your automatically generated ``parameters_model`` files, and you are all set to run POWDERDAY systematically for large numbers of galaxies in a snapshot!

