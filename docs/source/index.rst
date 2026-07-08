.. mphys-surrogate-model documentation master file, created by
   sphinx-quickstart on Mon Nov  3 09:32:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mphys-surrogate-model documentation
===================================
This repository contains python scripts for training latent-space machine learning representations of warm rain
microphysics processes. Superdroplet-enabled simulations provide high information-density training data upon which
various data-driven model structures are trained. All structures share in common a latent-space discovery based on
an autoencoder; differences lie in the varying representation of time-evolving dynamics within the latent space,
which utilize one of three model structures: (1) SINDy; (2) a neural-network derivative; (3) a finite-time step
autoregressor.

The **coalescence** representations in this repository correspond to the published study:

   de Jong, E. K., Gunawardena, N., Katona, J. E., Beydoun, H., Ghosh, D., & Caldwell, P. (2026).
   Data-Driven Reduced Order Modeling for Warm Rain Microphysics. *Journal of Geophysical Research: ML and Computation*.
   https://doi.org/10.1029/2025JH001103

Other microphysics processes, including **sedimentation**, are actively in development.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   training
   uq
   src



Indices
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`