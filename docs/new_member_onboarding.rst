New Member Onbarding
********************************


This doc goes through the main tools that are worth installing as a
new group member.  While you might not use all of these right away,
there's a pretty good chance you'll end up using most if not all of
them eventually.  


.. contents:: Section Contents 
    :local:

Slack Group
===================================================

We encourage heavy use of the Slack group.  In general if you have science-related questions, please post them in public channels for a few reasons:

#. A member of the group will likely get back to you faster than Desika can
#. There is almost always a member of the group that knows more about a given topic than Desika
#. The question can inspire discussion, motivate changes, and at the least, will be searchable for the future student/postdoc/colleague.

Please note a few additional items regarding professional behavior in this Slack Group:

#. The group includes a wide range of people, including active members in Gainesville, former members, collaborators/colleagues who are in industry, and collaborators/colleagues around the world.
#. One of the best parts about the slack group is the positive atmosphere, and encouragement when responding to help questions.
#. Please note that the slack group is an extension of your professional in-person life at UF, and should be treated as such (abiding by all UF rules and regulations, including a very strict no harassment policy).
#. This is a paid group, meaning messages are saved for posterity so that we can use the group as a resource going forward. 
#. Party Parrots always encouraged |P|

.. |P| image:: images/party.gif
    :width: 20


Hipergator
=====================

HiPerGator is our supercomputer, and you can find notes on usage here: 

https://desikasgroupofawesome.readthedocs.io/en/latest/intro_to_hpg.html  

In specific, see the Python Anaconda Environments section for installing and using a Conda environment.


Overview of Group Tools
============================

In our group, we use a wide variety of tools to run, analyze, and visualize our galaxy formation simulations.  Below we describe the most commonly used software in the Group:

#. Powderday: https://powderday.readthedocs.io/en/latest/.  (Installation notes here always supercede those on the desikasgroupofawesome.readthedocs page, though hopefully they're synced up.)
#. Caesar: https://caesar.readthedocs.io/en/latest/
#. yt: https://yt-project.org/



So what should you be doing?
============================

A good way to start is:

#. Make sure you can install and use a Conda environment
#. Install yt within this environment
#. Install caesar within this environment
#. Install powderday within this environment
#. Make sure you can run one of the example problems that ships with powderday on the HiPerGator cluster using the queue system (this last one will be much harder/more time consuming than 1-4).


Excercises
============================

#. Powderday: successfully run the gizmo_mw_zoom powderday run that ships with the code (in pd/tests/SKIRT)

#. Caesar: use a snapshot from a galaxy cosmological simulation on HiPerGator like: /orange/narayanan/desika.narayanan/gizmo runs/simba/m25n512/output/snapshot_305.hdf5 to:

   #. Create a Caesar file
   #. Plot the cumulative mass distribution functions for all of the dark matter halos in the snapshot, and the galaxies
   #. Use the above snapshot to plot a histogram of the gas fraction (Mgas/Mstellar)
   #. Use the above snapshot to plot a histogram of the dust to gas ratios of all the galaxies
