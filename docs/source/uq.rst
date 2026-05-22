Uncertainty Quantification (UQ)
================================

The ``UQ/`` directory provides tools for uncertainty quantification on trained coalescence surrogate models
using conformal prediction methods.

Overview
--------

Conformal prediction is a distribution-free method for constructing valid prediction intervals with
guaranteed coverage. This implementation provides split conformal prediction for:

* **Reconstruction uncertainty**: Confidence bands on autoencoder reconstructions
* **Latent dynamics uncertainty**: Confidence regions in latent space
* **End-to-end prediction uncertainty**: Full forward prediction intervals
* **Mass conservation**: Uncertainty quantification on total mass predictions

Core Scripts
------------

The UQ module consists of standalone scripts designed to be run from the command line:

**errors.py**
  Plots areas between prediction/confidence bands as a function of time for different
  network subsets (reconstruction, latent, end-to-end, mass).

**plot.py**
  Visualizes conformal prediction results at specified times and gridboxes for different
  network subsets.

Conformal Prediction Scripts
-----------------------------

The ``conformal/`` subdirectory contains model-specific conformal prediction implementations:

* ``ae_SINDy.py`` - Conformal prediction for AE-SINDy models
* ``ae_NNdzdt.py`` - Conformal prediction for AE-NNdzdt models
* ``ae_AR.py`` - Conformal prediction for AE-AR models
* ``cp_test.py`` - Testing and validation utilities
* ``extract_test_idx.py`` - Test set index extraction

Usage
-----

Basic Workflow
~~~~~~~~~~~~~~

1. **Train a model** using scripts in ``training_scripts/``
2. **Calibrate conformal predictor** on held-out calibration data:

.. code-block:: bash

   cd UQ/conformal
   python ae_SINDy.py dataset_name -p 20 -a 0.1

   # -p: percentage of data for calibration (default 20%)
   # -a: miscoverage rate (alpha), default 0.1 for 90% coverage

3. **Visualize prediction intervals**:

.. code-block:: bash

   cd UQ
   python plot.py dataset_name -s nomass -p 20

   # -s: subset to plot (nomass or all)
   # -p: calibration percentage used

4. **Plot prediction bands**:

.. code-block:: bash

   cd UQ
   python errors.py dataset_name -s decoder -t "0 5 10" -p 20

   # -s: subset (decoder, full, or mass)
   # -t: time indices to plot

Parameters
~~~~~~~~~~

**Calibration percentage (-p)**:
  Percentage of data used for calibration (default 20%). Higher values give tighter intervals
  but require more data.

**Miscoverage rate (-a, alpha)**:
  Target miscoverage probability (default 0.1 for 90% coverage). The method guarantees that
  at least (1-alpha)×100% of true values fall within the prediction intervals.

**Subset options**:

  * ``decoder`` - Reconstruction/autoencoder only
  * ``full`` - Entire network (reconstruction + dynamics)
  * ``mass`` - Mass conservation predictions
  * ``nomass`` - All except mass (3×1 layout)
  * ``all`` - Everything (2×2 grid layout)

Mathematical Background
-----------------------

Split Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a pretrained model :math:`\hat{f}`, conformal prediction constructs prediction sets
:math:`C(X_{test})` that satisfy:

.. math::

   P(Y_{test} \in C(X_{test})) \geq 1 - \alpha

The method:

1. Split data into training (already used), calibration, and test sets
2. Compute nonconformity scores on calibration set: :math:`s_i = |Y_i - \hat{f}(X_i)|`
3. Find quantile: :math:`\hat{q} = \text{Quantile}(\{s_i\}, (1-\alpha)(1 + 1/n_{cal}))`
4. Construct intervals: :math:`C(X) = [\hat{f}(X) - \hat{q}, \hat{f}(X) + \hat{q}]`

For multivariate outputs (droplet size distributions), we use:

* **Mahalanobis distance** for latent space confidence regions
* **Pointwise intervals** for reconstructions and full predictions

Output Files
------------

Results are saved to ``UQ/conformal/results/`` and ``UQ/logs/`` as pickle files containing:

* Calibrated quantiles for each uncertainty component
* Prediction intervals on test data
* Coverage diagnostics
* Nonconformity scores

Related Work
------------

This implementation is based on:

* Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*
* Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
  "Distribution-free predictive inference for regression." *JASA*
* Angelopoulos, A. N., & Bates, S. (2021). "A gentle introduction to conformal prediction
  and distribution-free uncertainty quantification." arXiv:2107.07511

Batch Processing
----------------

Shell scripts are provided for running conformal prediction on multiple datasets:

.. code-block:: bash

   cd UQ/conformal

   # Run on all models for a dataset
   bash ae_SINDy.sh
   bash ae_NNdzdt.sh
   bash ae_AR.sh

   # Generate plots
   bash plotting.sh

See Also
--------

* :doc:`training` - Model training documentation
* :doc:`src.diagnostics` - Diagnostic metrics for model evaluation
