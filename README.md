## GravCHAW

## What is GravCHAW?

GravCHAW is an open-source Python software for the assimilation of time-lapse gravity (TLG) data into numerical groundwater models. It incorporates a site-independent coupled hydrogravimetric model that links FloPy, a Python interface for MODFLOW 
([Bakker et al., 2016](https://doi.org/10.1111/gwat.12413); [Hughes et al., 2024](https://doi.org/10.1111/gwat.13327)), to Gravi4GW-hybrid, a newly developed Python module for simulating TLG data. The coupled model is further integrated with pyEMU ([White et al., 2016](https://doi.org/10.1016/j.envsoft.2016.08.017)), a Python interface for the PEST++ software suite.


## What does it do?

The main functionality of the framework is assimilating time-lapse gravity data to estimate hydraulic parameters, make predictions of state variables, and quantify their associated uncertainties. These tasks are carried out using a suite of optimization and uncertainty analysis algorithms in a Bayesian context, integrated within the framework. GravCHAW can perform a coupled hydrogravimetric inversion assimilating TLg data individually or jointly with hydrological observations.

## How to cite it

Mohammadi, Nazanin, Hamzeh Mohammadigheymasi, and Landon JS Halloran. "GravCHAW: A software framework for the assimilation of time-lapse gravimetry data in groundwater models." Computers & Geosciences (2026): 106118. https://doi.org/10.1016/j.cageo.2026.106118



## Installatin instructions

1. Download the repository as a zip file from here: https://github.com/GreenResModler/gravchaw.git.

2. Unzip the folder to a desired location on your machine.

3. Ensure Python is installed via Anaconda, if not install https://www.anaconda.com/.

4. On Windows, open the Anaconda Prompt from the Start Menu. On Linux, simply use your regular terminal.

5. Navigate to the unziped folder directory. The environment.yml file is located in this folder.

6. Run "conda env create -f environment.yml" to install all dependencies.

7. Now an anaconda environment,"tlg-gw-assim" is created on your machine (you may change the environment name by editing the "name" field in the "environment.yml" file before running step 6).


## How to use it?

Activate the created environmnet by running "conda activate software_paper" (or if you used a custom name "conda activate your_desired_env_name").

Jupyter notebooks in the repository guide you how to use the framework. To start Jupyter Notebook, make sure you are in the unziped folder directory in your terminal, then run "jupyter notebook".

Note: Each time you start a fresh session (e.g., open Anaconda Prompt on Windows or a terminal on Linux), remember to:

1. Activate the environment

2. Navigate to the unzipped folder directory before launching Jupyter notebook.


The Jupyter notebooks are located in the "example" folder. You can navigate to this folder using the Jupyter notebook file browser once the notebook server starts (ensure your default web browser opens automatically). 

To familiarize yourself with performing GravCHAW, start by running the notebooks in the following subfolders:

"01_create_gw": Running the notebook in this folder is required before running other notebooks in the repository. It creates the groundwater model and guides you through the necessary settings to perform the next steps of the framework.

Notebooks in the following subfolders perform parameter estimation, make predictions, and quantify uncertainty. These notebooks can be run independently, with the main difference being the type of observations they assimilate.

 "02_paper_case2": Performs the couplded inversion assimilating TLG data to estimate hydraulic parameters, predicte the state variable (hydraulic head)  and quantify parameter and prediction uncertainties (case 2 as described in the published paper).

 "03_paper_case3": Performs the couplded inversion assimilating TLG and multiple hydraulic data to estimate hydraulic parameters, predicte the state variable (hydraulic head)  and quantify parameter and prediction uncertainties (case 3 as described in the published paper).

 "04_paper_case4": Performs the couplded inversion assimilating TLG and one hydraulic data to estimate hydraulic parameters, predicte the state variable (hydraulic head)  and quantify parameter and prediction uncertainties (case 4 as described in the published paper).

 "05_paper_case1": Performs inversion of multiple hydraulic head data to estimate hydraulic parameters, predicte the state variable (hydraulic head)  and quantify parameter and prediction uncertainties (case 1 as described in the published paper).
 "06_objective_response_surface": Evaluates objective function response surface over paramter space.
 
 Review the notes within the notebooks to understand the necessary settings for each case.
 
The "observation_data" and "lowSNR_observation_data" contain the same observation types, differing only in the signal-to-noise ratio (SNR) of TLG data.

The "bin_new" folder, the "herebedragons.py" file, and the "mdf_response_surface.py" file are directly sourced from the pyEMU repository, providing executables within the framework’s working directory. 



