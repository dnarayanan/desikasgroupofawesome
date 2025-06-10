Common Software Tools and Procedures
**************************************
.. contents:: Section Contents
    :local:

Anaconda Environments
======================

Using new anaconda environments for individual projects or major code
initiatives can be super helpful, especially when testing and
debugging.  In what follows, we outline how to go about this.

After installing Anaconda, you can actiavte a new environment (and activate it) via::

  conda create --name myenv python=3.9
  conda activate myenv



When you do this, you are in your new environment, and any package you
install via conda will remain local to this conda environment, and not
see (or be touched) by any other environment.  Why would you want to
do this?

#. You can then delete an entire environment, and all of it's packages, without messing up other environments via::

     conda remove --name myenv --all


#. Imagine you're running a ton of (e.g.,) powderday simulations, and
   you want to code some tests up in a new branch of powderday and
   debug them without messing up your major simulation campaign you
   have going on.  Then you can install the new powderday test branch
   to a new environment, and run locally there, without worrying about
   your other environments getting messed up.
		
You can find many other options/nuances at
https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment
, though the above should be enough to get you going.


Jupyter Notebook
============		

If you’re used to doing analysis in a notebook or GUI setting, it’s a bit tricky to work on HPG.
Luckily it’s super easy to run Jupyter on HPG and access the notebooks on your local machine.
Here’s how to do it, just note to replace instances of my username (s.lower) with yours wherever it shows up:


On HPG, submit the job script below. The advantage of submitting this as a SLURM
script vs. just submitting as an inline job on some interactive (i.e., non login node) cores
is the notebook will stay open for up to 30 days!::

  #!/bin/bash
  #SBATCH --job-name=jnb
  #SBATCH --output=jnb.log
  #SBATCH --mail-type=ALL
  #SBATCH --mail-user={your email here}
  #SBATCH --time=30-00:00:00
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=4
  #SBATCH --nodes=1
  #SBATCH --mem-per-cpu=3900mb
  #SBATCH --qos=narayanan
  #if you use a conda env, load that here
  #source activate conda_env
  jupyter notebook --no-browser --port=8080

Save this script in a file called jnb.job, and then run it using 'sbatch jnb.job'. This script runs a jupyter notebook on HPG occupying 4 CPUs, but does not open a browser (cause HPG doesn’t have that ability). 

Once this notebook is up and running do::

  [s.lower@login3 ~]: squeue -u s.lower

  JOBID PARTITION NAME USER ST TIME NODES NODELIST
  2497853 hpg-milan jnb s.lower R 5:30:15 1 c0710a-s1
  
From this, make note of which node the job is located on. Here it's on ``c0710a-s1``. 

Then, in **your local terminal**, enter::

  ssh -N -L 8080:NODENAME.ufhpc:8080 s.lower@hpg2.rc.ufl.edu

where ``NODENAME`` will be the name of the node above. Lastly, open whatever preferred browser you use and enter in the address bar::

  localhost:8080

Which will open our ssh tunnel to the Jupyter notebook hosted on HPG. If this is the first time accessing that notebook, it will ask you for a token, which you can find in the ``jnb.log`` file.


You’ll have to do all the above shenanigans (besides debug related stuff) to start your jupyter notebook the first time (or to restart it after it runs out of time after 30 days). But during the
next 30 days, you’ll just start from step 3 to connect to the notebook from your computer.

Debugging Jupyter Notebook
------------

**Channel Open Failed**

If your notebook does not open in your browser and your terminals says something like::

  channel 2: open failed: connect failed: Connection refused

You may need to fix your jupyter config script. You can do that by copying mine over to your jupyter directory::

  cp /home/s.lower/.jupyter/jupyter_notebook_config.py $HOME/.jupyter

Once copied, you’ll need to cancel the jupyter notebook job and start a new one for it to recognize the config file.


**Internal Service Error**

This can occur when trying to open a python notebook from the localhost:8080. When this happens it is best to first try updating conda with::

  conda update --all

Then you can continue by using pip to upgrade jupyter::

  pip install jupyter --upgrade

If this doesn’t work, try::

  conda install -c conda-forge jupyter_contrib_nbextensions

Hopefully one of these fixes the issue.


Tmux
============
Tmux stands for the terminal multiplexer. Don't worry about the fancy terminology. The main use of tmux for us at least is to keep the single/multiple terminal sessions running even after you log out from hipergator. To load tmux just do::

    	module load tmux

and to start a tmux session just type:: 

    	tmux
    
This will start a tmux session which will look like a new terminal window. You can run your codes in this window and they will keep running even after you log out from hipergator. Once you have your code running, you can exit the tmux window by hitting::

  	ctrl–b–d
	
To list all the tmux sessions you have runnning use tmux ls and to open an existing tmux session use:: 

	tmux attach -t session-name


Keep in mind that you cannot scroll by default inside a tmmux session. To scroll you must hit::
	
	ctrl–b–[
	
to go into scrolling mode. To get out of it just hit *q*
	
A list of the most useful tmux commands can be found here: https://danielmiessler.com/study/tmux/

One common use for tmux sessions is to ask for a development node inside a tmux session. You can do so via::

	srun --pty --partition=hpg-dev --time=10:00:00 --nodes=1 --ntasks=10  --mem=60gb --qos=narayanan bash -i
	
Once the node is allocated you can close the tmux session and then you can access this node anywhere using::
	
	ssh -XC \$(squeue -u $USER -n bash --states=R --noheader --Format=nodelist | head -n1)
	
It is recommended to alias the above two commands in your .bashrc file. This can be done by adding::

	alias devnode="ssh -XC \$(squeue -u $USER -n bash --states=R --noheader --Format=nodelist | head -n1)"
	alias dev='srun --pty --partition=hpg-dev --time=10:00:00 --nodes=1 --ntasks=10  --mem=60gb --qos=narayanan bash -i'
	
to your .bashrc file which can be found in your home directory. Once you have edited the bashrc file type::

	source .bashrc
	
to reload the .bashrc file and now you should be able access these commands just by using their aliases. This can be done for any commands you frequently use.

Git (todo)
============

.. include:: vs-code.rst

