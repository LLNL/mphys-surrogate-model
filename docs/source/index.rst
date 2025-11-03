.. mphys-surrogate-model documentation master file, created by
   sphinx-quickstart on Mon Nov  3 09:32:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mphys-surrogate-model documentation
===================================
This repository contains python scripts for training latent-space machine learning representations of warm rain
droplet coalescence and is the companion to a paper currently in review, titled
"Data-Driven Reduced Order Modeling for Warm Rain Microphysics". Superdroplet-enabled simulations of this warm
rain formation process provide high information-density training data upon which various data-driven model
structures are trained. All structures share in common a latent-space discovery based on an autoencoder (1);
differences lie in the varying representation of time-evolving dynamics within the latent space, which utilize
one of three model structures: (1) SINDy (2); (2) a neural-network derivative; (3) a finite-time step autoregressor.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   src



Indices
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`