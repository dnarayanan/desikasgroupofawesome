Using the CAMELS Simulation Suite for Analysis
**********
.. contents:: Section Contents
    :local:
    
What is CAMELS?
============

The CAMELS (Cosmology and Astrophysics with MachinE Learning Simulations) project is a vast project based at the Center for Computational Astrophysics (CCA). In total, it comprises >14,000 cosmological simulations. The important features of CAMELS are (1) that CAMELS represents many different cosmological models (at the time of writing, they have boxes run with IllustrisTNG, SIMBA, Astrid, Magneticum, and EAGLE physics) and (2) that the CAMELS team has run boxes of each of these simulation models with their physical parameters varied in different ways. CAMELS seeks to explore the constraints on cosmology and astrophysics we can derive using this vast set of data available, particularly with a focus on applying it to machine learning models. A natural advantage of CAMELS for us theorists is that using different CAMELS simulations can allow you to not be tied to any particular set of simulation physics nor be tied to the specific set of parameters that were used to produce the fiducial runs for many of these simulations. An important caveat to keep in mind when using CAMELS, however, is that to make running all of these simulations feasible, CAMELS boxes were run at lower resolution and with smaller box sizes - for SIMBA, it is roughly like the mass resolution of the fiducial m100n1024 box at the size of the m25 box. This can limit the mass ranges of galaxies that are produced by the simulations. For more details on CAMELS, read through their documentation page (https://camels.readthedocs.io/en/latest/index.html).


What data are available through CAMELS?
============
The CAMELS simulation suite offers many data products to analyze. I'm going to give a brief overview of the data products that you are likely to use, but check out the documentation for full details (https://camels.readthedocs.io/en/latest/index.html).

All CAMELS data is available through GLOBUS (more on that later). First, the raw cosmological boxes that the CAMELS team has run are accessible - most of the simulations have full snapshots available from to z=6 up to z=0. Associated with these boxes are galaxy/halo catalogs produced from various different codes' Friends-of-Friends (FOF) algorithms (CAESAR, SUBFIND, ROCKSTAR). Note that, at the time of writing, many CAESAR catalogs are missing for more recently run simulations. Some cosmology quantities (such as the power spectra) are also already available. It's also probably worth loading https://github.com/DhruvZ/dtz_camels_workflow_scripts as many of the explanations below have some sort of example in this page and it is a self-contained workflow for using a set of CAMELS results.



Globus 
============
Globus is data transfer tool that is connected to many datasets. Globus is one way to access CAMELS data - the CAMELS data is available through Globus and Hipergator is nicely set up to use Globus. Follow the UFRC HPG introductory documents for Globus if you are a first-time user and to get a sense of its uses (https://help.rc.ufl.edu/doc/Globus). You'll want to make sure you can log in, and I'd recommend you create your own guest collection following the instructions on the above intro page to whatever is the appropriate home directory for it on HPG. I'd also recommend you look up the CAMELS collection and familiarize yourself with the file structures and where everything you might want is available in the full CAMELS suite.


Simple Globus data loading
-----------------
Once you're logged in, Globus has a fairly straightforward online interface to get the data you want. First, search for the 'CAMELS' collection, and it should be one of the first results. Next, selecting the 'Transfer or sync to' option will open up a second panel which you'll then want to fill in with the collection you made earlier onto the search bar. Finally, navigate and select the files on CAMELS you want and the output directory on HPG and click the 'Start' button on the CAMELS side. This will start the data transfer, which you can monitor with the 'ACTIVITY' tab on the left of the website.


Inline Globus data loading
-----------------
The previous explanation discussed how you can take advantage of the intuitive setup of Globus to access CAMELS simulation results. However, it is possible that you might want to load many of these simulations simultaneously, access different parts of the original dataset, or simply just not have to leave command line in HPG to accomplish this. Globus actually does have an API for inline transfers (https://docs.globus.org/cli/reference/transfer/). You'll have to ``module load globus`` in HPG before you can use this API. The basic command structure (as shown in the above link) follows ``globus transfer [OPTIONS] SOURCE_ENDPOINT_ID[:SOURCE_PATH] DEST_ENDPOINT_ID[:DEST_PATH]``, where the IDs are hexadecimal IDs associated with Globus collections, one for CAMELS and one for your local connection. The paths refer to the paths from the root folder in both collections where you want the data to be loaded from and sent to. Personally, I recommend using a bash script to do this in a more automated and consistent way. For a reference, it might be worth starting with https://github.com/DhruvZ/dtz_camels_workflow_scripts/blob/master/camels_single_workflow/camels_bash_sim_load.sh. This is a bash script meant to load the Caesar, SUBFIND, and full simulation to different locations in the collection for a particular '1P' (the team has only varied one simulation parameter) run as pulled from a reference text file with a confirmation prompt before the actual request is sent to the Globus system. This is by no means a comprehensive script to do this and will need to be modified for your collection ID and paths, but this is a good starting point to go from that can likely be molded into what you need for your individual problem.



Filtering galaxies (again)
============

At this point, you hopefully have a loaded simulation+SUBFIND/CAESAR files depending on the simulation type. As we often do, it makes sense to try to filter these if you are running radiative transfer down the line. Again, the script at https://github.com/DhruvZ/dtz_camels_workflow_scripts/blob/master/camels_single_workflow/camels_ml_pd_setup.sh and its dependencies could serve as a starting basis for what to call for filtering. Below, I'll touch on running filtering for TNG using CAMELS from the SUBFIND catalogs since that differs noticeably from the traditional Caesar-based SIMBA filtering that we often do, and the CAMELS TNG sims and catalogs are a litte different in structure than the flagship TNG results.

Filtering for CAMELS-TNG results
-----------------
Nominally, we could filter any simulation using either the Caesar or SUBFIND catalogs. SUBFIND has a couple important differences from Caesar, pretty much all of which can make our life more annoying. The first difference is the organization. When TNG is run, its outputs are sorted by the halo and subhalo that particles are associated with. As a consequence, the SUBFIND catalogs, rather than storting particle ID lists the same way that Caesar would, simply list the number of particles that are associated with each halo and subhalo. There is also a 'free agent' section of each halo for particles in the halo but not in any of its subhaloes. Therefore, to filter, you need to find the starting index of the subhalo by (1) adding up the lengths of all halos that come before the one that this subhalo is a part of for each particle type and (2) then add up the lengths of all previous subhalos that in the same halo as the current subhalo; after, you need to cut out all particles in range from the starting index to the length for the particle types. This would be made easier if the offset files for these subhalos, which are a normal TNG data product, were available for the CAMELS runs, but they are not. Furthermore, SUBFIND doesn't have lists of 'galaxies' the same way that Caesar does. It only has lists of subhalos, which may or may not have stellar mass associated with them. Therefore, when we filter, we also need to apply some kind of relevant split for galaxies with stellar mass and likely record this, as we cannot run radiative transfer on galaxies with no star particles. Additionally, TNG associates wind particles with PartType4 (the same as stars), so there is potential for confusion there, especially as SUBFIND counts the wind particles with stars and gas differently depending on the quantity.

All this would seem like a convincing argument to just use Caesar to filter. Unfortunately, Caesar causes problems with TNG specifically in several ways. Most notably, there is a fix in POWDERDAY at the time of writing for AREPO snapshots that any star particles outside of the gas grid are deleted to prevent a crash. We have noticed as well with testing that the Caesar catalogs made for TNG runs seem to have low amounts of gas, potentially because of the Caesar density cut. This resulted in tests where >33% of the star mass was removed from the POWDERDAY runs. Therefore, it is recommended to use the SUBFIND catalogs to filter.

https://github.com/DhruvZ/dtz_camels_workflow_scripts/blob/master/filter_tng_camels_setup.py serves as a blueprint to perform this filtering.
