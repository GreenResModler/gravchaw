
GravCHAW

What is GravCHAW?
-----------------

GravCHAW is a Python software to assimilate time-lapse gravity in numerical groundwater models. It integrates newly developed time-lapse gravity data simulation module to FloPy (Langevin et al., 2017), forming coupled hydrogravimetric model.
The coupled model is integrated with pyEMU (White et al., 2016).


What does it do?
----------------

The main functionality of the framework is assimilating time-lapse gravity data to estimate hyrogeological parameters, make predictions, and quantify their associated uncertainties. These tasks are carried out using a suite of optimization and uncertainty quantification algorithms integrated within the framework.


Installatin instructions
------------------------

1. Download the repository as a zip file from here:https://github.com/GreenResModler/gw_unc_reduction.git.

2. Unzip the folder to a desired location on your machine.

3. Ensure Python is installed via Anaconda, if not install https://www.anaconda.com/.

4. On Windows, open the Anaconda Prompt from the Start Menu. On Linux, simply use your regular terminal.

5. Navigate to the unziped folder directory. The environment.yml file is located in this folder.

6. Run "conda env create -f environment.yml" to install all dependencies.

7. Now an anaconda environment,"tlg-gw-assim" is created on your machine (you may change the environment name by editing the "name" field in the "environment.yml" file before running step 6).


How to use it?
--------------
Activate the created environmnet by running "conda activate software_paper" (or if you used a custom name "conda activate your_desired_env_name").

Jupyter notebooks in the repository guide you how to use the framework. To start Jupyter Notebook, make sure you are in the unziped folder directory in your terminal, then run "jupyter notebook".

Note: Each time you start a fresh session (e.g., open Anaconda Prompt on Windows or a terminal on Linux), remember to:

1. Activate the environment

2. Navigate to the unzipped folder directory before launching Jupyter notebook.


The Jupyter notebooks are located in the "example" folder. You can navigate to this folder using the Jupyter notebook file browser once the notebook server starts (ensure your default web browser opens automatically). 

To familiarize yourself with performing GravCHAW, start by running the notebooks in the following subfolders:

"01_create_gw": Running the notebook in this folder is required before running other notebooks in the repository. It creates the groundwater model and guides you through the necessary settings to perform the next steps of the framework.

Notebooks in the following subfolders perform parameter estimation, make predictions, and quantify uncertainty. These notebooks can be run independentlyThese notebooks can be run independently, with the main difference being the type of observations they assimilate.

 "02_paper_case2": Performs inversion for Case 2 as described in the manuscript.

 "03_paper_case4": Performs inversion for Case 4 as described in the manuscript.

 "04_paper_case3": Performs inversion for Case 3 as described in the manuscript.

 "05_paper_case1": Performs inversion for Case 1 as described in the manuscript.
 
 Review the notes within the notebooks to understand the necessary settings for each case
 
The "observation_data" and "lowSNR_observation_data" contain the same observation types, differing only in the signal-to-noise ratio (SNR) of TLG data.

The "bin_new" folder, the "herebedragons.py" file, and the "mdf_response_surface.py" file are directly sourced from the pyEMU repository, providing executables within the framework’s working directory. 

The corresponding manuscript related to this software is submitted to the journal "Computers and Geosciences".
