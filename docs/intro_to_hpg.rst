Getting Started with Hipergator
********************************

The HiPerGator supercomputing cluster (abbreviated HPG from here on out) is our main platform for computing and sharing files. We highly encourage reading through the documentation here: `<https://help.rc.ufl.edu/doc/Training.>`_ Below we'll detail specific aspects of using hpg within our group, starting with accessing the cluster from your local computer and navigating around the various group allocations to running code and using the job scheduler.

.. contents:: Section Contents 
    :local:

But first, a few things to note about working in a cluster environment. 

General Things About Working in a HPC Environment
===================================================

High performance computing (HPC) environments are a bit different than working locally or on a smaller server/cluster and can take a while to get used to. There's two important things to keep in mind when working with HPG

  1. HPG relies on a 'module load' system to load libraries like compilers and other /root installed software
  2. Any work done on HPG cannot be run on a login node.


To elaborate on the first point, HPG has a large set of software installed that is accessible to all users through the module load system. In short, this allows many conflicting libraries, e.g. intel vs. gcc compilers, to be installed at the same time but only loaded when necessary. The full (and updated?) list of modules available can be found here <https://help.rc.ufl.edu/doc/Applications> along with a more exhaustive set of instructions on how to use the module system. We'll show an example of how to do this in the "Submitting SLURM Jobs" section. 

The second point is maybe even more important as it dictates most of our workflow on HPG. As we'll explain below, when you log into HPG, you land in a login node. From the login node, like every other node, you can access all of your data. But keep in mind that no code should be run via a login node. To run code there's two (or three, if you use Jupyter notebooks) options depending on what type of work you're doing. All of these options rely on the internal job sche\
duler that is in charge of allocating users' jobs to the CPUs available. This scheduler is called SLURM and thankfully has pretty useful documentation, so we recommend reading up on it <https://slurm.schedmd.com/documentation.html>. The basics of SLURM involve commands like 'srun' in combination with different keywords that specify how long you want to use the CPUs you're requesting, what HPG partition you need, and which allocation you want to schedule your job through (see the next section for details on our group's allocations). We'll also go over how to interface with SLURM and submit jobs in the "Submitting SLURM Jobs" section.


Accessing Hipergator
=====================

To access hipergator from your computer, open a terminal and enter::

  ssh hpg2.rc.ufl.edu gatorlinkid

This will log you into hipergator and set you on a 'login node.' As stated above, you should not run any code on a login node. But you can do any other things like navigate through your directories, write small data like python scripts or text files, etc. Anything that takes CPU power/IO will need to be done on a compute node. We'll talk about how to get to one of those later. 


Running Code on Hipergator
============================

Python Anaconda Environments
------------------------------

Generally it's advised to work within individual Anaconda python
environments on HiPerGator.  The reason is that this allows you to isolate individual code builds, and if something gets screwed up, you can just delete the environment without screwing anything up. The following is taken from:

https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
https://help.rc.ufl.edu/doc/Conda

To start a new environment::

  module load conda
  mamba create --p /blue/narayanan/desika.narayanan/conda/envs/myenv python=3.8

this will install the environment in your /blue data directory.  This
is very important since otherwise these conda environments will blow
up in space, and eventually use all your home disk space if you
install it in $HOME.  You can then activate it via::

  ml conda
  mamba activate /blue/narayanan/desika.narayanan/conda/envs/myenv

which of course you might prefer to put in your .bashrc file as an alias.  For example, mine looks like this::

  alias py38='ml conda; mamba activate /blue/narayanan/desika.narayanan/conda/envs/py38

so that at the linux prompt, I just have to type::

  py38

And it will automagically load my python3.8 environment (note, right
away we can see the potential of having different environments -- I
can have different python versions for example, as I test code out, etc.).



Cluster, Nodes, Partitions, and CPUs
-----------------

The HPG cluster is physically comprised of a bunch of ’nodes,’ which house cores/CPUs -- I (Sidney) use these terms interchangeably which probably isn’t technically correct but it will work for our purposes. For nodes in the HPG2 (hipergator2.0) allocation, there are 32 cores per node. For HPG3 there are 64 cores in the default nodes. Each core has a memory allocation of 4GB/8GB for HPG2 and HPG3, respectively. If we compare to a regular desktop computer, that represents a single node, and depending on how sophisticated your CPU hardware is, it can have from 2 to 8 cores. There are several classes of nodes (called partitions) on HPG, depending on what version of HPG they are from (literally, when were they installed) as well as what kind of CPUs each partition has. See here <https://help.rc.ufl.edu/doc/Available_Node_Features> for the full list of hipergator partitions and the details about the CPUs and memory resources for each.


Queues
-----------------

To run code, we use the nodes/CPUs described above. But we (unfotunately) don't have access to all CPUs on HPG at once. Our group has priority access to 950 CPUs and shared access (i.e. we have to wait in line for it) to 8550 CPUs. These two "queues" (known as Quality of Service [QOS] in HPG/SLURM lingo) are called ``narayanan`` and ``narayanan-b``.  The former represents our investment queue and the latter is our burst queue. You'll specify which queue to submit your jobs to in your SLURM command/script, which we'll go over shortly. 

The trade offs of the two queues come from how to use each. The investment queue is like fast pass at Disney World: time to get requested cores is much shorter than burst and we can occupy them for much longer, but we have a limited number of cores (for the record, I've never actually gotten a fast pass at the one time I was at Disney World so maybe this analogy falls flat). The time we can use the cores is dependent on which allocation we request them from: we can use investment queue cores for 30 days while burst queue jobs can only run for 4 days. 

In general, the rule of thumb is that investment cores will start much more quickly than burst cores, but are of course more limited.  We suggest using investment sparingly: for getting interactive/debugging jobs, or for small jobs that are being tested that need to be turned around relatively quickly for debugging.


If you need more than ~100 investment cores, please check in the #general channel in slack to see if it's okay.


Storage Space
-----------------

There are three directories that you have access to, regardless of what node/partition/queue you're accessing them from::

  1. /home/your_gator_login_name
  2. /blue/narayanan/your_gator_login_name
  3. /orange/narayanan/your_gator_login_name


/home/yourname is your home directory.  This is backed up, and has a relatively low storage limit (~40 GB).  This is meant for source code, but not really data. The latter two drives are for data. In general, /blue reads/writes faster and is meant for active simulations that are running, while /orange is meant for more long term storage.  This said, we've noticed relatively little difference between the two as far as performance goes.

Submitting SLURM Jobs
-----------------

The first option for running code when working on HPG is to use what we call an interactive node, which means we request some number of CPUs/cores from computing nodes to work 'interactively.' This is in contrast to submitting your code as a job to the job scheduler, which we'll get to in a second. Doing things interactively is essentially like doing command line work on your local machine. We just have to do a couple of steps in between because work cannot be done on login nodes. A commong way to do this is by requesting CPUs from the dev partition, which are short-term access (> 12 hours) CPUs that typically have very little demand, so you can access them relatively quickly compared to other nodes. To access a dev node to do work in, run this command::

  srun --pty --partition=hpg-dev --qos=narayanan --time=8:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 -u bash -i

which requests 16 cores on a dev node for 8 hours through the investment queue. This should give you access to a dev node within seconds and you can get to working. It is also really helpful to set some commonly used commands like this in your ``.bashrc`` file as aliases for ease of access. You can find your bash file at ``/home/your_gator_login_name/.bashrc`` and set an alias by doing::

  alias interact='srun --pty --partition=hpg-dev --qos=narayanan --time=8:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 -u bash -i'

The second way to run code is to schedule a job using the 'sbatch' command. The easiest way to explain this is by showing an example sbatch script (saved as a text file with some name like python_script.job)::

    #!/bin/bash
    #SBATCH --job-name=example_script 
    #SBATCH --output=example_script.log 
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=__ your email here __
    #SBATCH --time=01:00:00 
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --nodes=1
    #SBATCH --mem-per-cpu=3900mb
    #SBATCH --qos=narayanan-b
    
    python some_python_script.py

This bash script is how we communicate with slurm to run jobs we don't need to handle interactively. Here, we are asking to run a python script, which we know needs up to an hour to run, on 4 CPUs in the burst queue. We are also specifying the memory needed at 3900mb. Of course, these parameters can be adjusted to accommodate whatever code you are running. For example, typical runs with our ``powderday`` code take about 30 minutes to run, so I'd be safe and request 1.5 hours for the ``--time`` parameter. 

One thing to remember is that HPG uses a module load system to load root installed things like compilers. So if your code requires such software -- say, an intel compiler and MPI -- you can load it during on an interactive node like::

  [s.lower@login3 ~]: module load intel/2018.1.163 openmpi/4.0.3

If using a job script, put the ``module load`` statement in the job script before the line running your code. 

To submit the job script for the code above, run the command::

  sbatch python_script.job


This will send the job to SLURM, which will figure out where/when this request can fit in with everyone else's job requests. In contrast to running code interactively, submitting jobs to SLURM means your code will run completely remotely (i.e., once you submit, you don't have to stay on HPG until it finishes).

Checking SLURM Job Status
-----------------

To check the status of any jobs you have currently in queue, you can run the command::

  squeue -u your_gator_login_name

which will display all jobs submitted to queue, either running or awaiting allocation, separated by which queue (investment or burst) you submitted to. You can also use the command 'slurmInfo -u narayanan' to check the entire group's cumulative CPU usage, but you'll need to module load ``ufrc`` beforehand.

You can see what jobs you (and the entire group) have in queue, and what QOS they've been submitted to using::

  showq -A narayanan

If you find that your jobs are sitting in queue for longer than you
expect, and not getting anywhere, check the 'reason' against this
glossary: https://help.rc.ufl.edu/doc/Why_is_my_job_not_running


