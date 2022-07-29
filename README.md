# Wind erosion model calibration 

This repository contains the code base for calibrating a model for wind erosion in the presence of vegetation introduced in [Okin (2008)](https://www.researchgate.net/publication/248805022_A_new_model_of_wind_erosion_in_the_presence_of_vegetation). Building on the calibration methods outlined in [Li et al. (2013)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/jgrf.20040), we generate a set of best-fit parameters for the model using *in situ* and remote sensing data. The calibrated model will be used to generate high resolution estimates of sediment flux for the state of California using a machine learning algorithm detailed in [Zhou et al. (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0034425719305401).<br><br>
Our calibration relies on sediment flux measurements from the [National Wind Erosion Research Network](https://winderosionnetwork.org/), from both ecological and agricultural sites, along with wind speed measurements. In our calibration, we've tested the method with both weather station data and NARR wind reanalysis data (3 hourly 10m windspeeds). We use remote sensing-derived data-- vegetation heights for several vegetation types and a probability density function of gap sizes-- from the Zhou machine learning algorithm to characterize the vegetation at each site, used in the model as a representation of scale height (horizontal distance / vertical distance).<br><br>
The calibration method utilizes an iterative process to generate a set of 1000 random values for each parameter within a defined range. It then solves for the set of parameters that give lowest root mean squared error of the logs (RMSEL) between the modelled and measured sediment flux. This process is perfomed for each of the sites following the leave-one-out cross validation method described in Li et al. (2013) in which one site is left out of the error minimization, allowing for RMSEL to be computed using flux predicted for the leftout site using the set of best-fit parameters derived from the remaining sites. Using this method, RMSEL is computed using the log of the predicted flux of the leftout site minute the log of the actual flux of the leftout site. For example, if you're performing the calibration with 8 sites, the code will run 8000 iterations (8 sites * 1000 iterations per site) to generate a set of best-fit parameters for the model. 

## Contact 
**Nicole Keeney (code author)**<br>
UC Berkeley School of Public Health, Department of Environmental Health Sciences<br>
nicolejkeeney@gmail.com<br><br>
**Abinash Bhattachan (study lead)**<br>
California State University East Bay, Department of Earth & Environmental Sciences<br>
abinash.bhattachan@csueastbay.edu

## Model parameters 
Below, we provide a description of the parameters derived in the model calibration and the symbols used to reference them in the code. 
 - z0: roughness length (meters)
 - A: constant (variable units) 
 - C: e-folding distance for recovery of the shear stress in the lee of plants (dimensionless)
 - U: shear velocity ratio in the immediate lee of plants; the ratio of (shear velocity downwind of a plant)/(shear velocity) at x=0 meters of horizontal distance (dimensionless)<br>

We also solve for the best size to use for the largest gap size bin (called `max_gap` or `max_gap_bin` in the code) and the best Q flux equation to use. 

## Measurement sites 
We've used sediment flux data from the following sites: 
 - Pullman: MWH, Moses Lake/Grant County
 - Moab: CNY, Moab/Canyonlands
 - Holloman: HMN, Holloman AFB
 - Mandan: Y19, Mandam
 - San Luis Valley: ALS, Alamosa Municipal County Airport
 - NEAT 1, 2, & 3: Las Cruces, New Mexico

## Repository structure
 - `calibration.py`: The model calibration script. Allows the user to specify the number of iterations to run and which set of u threshold values to use for each site.
 - `WEMO_model.py`: Code for computing sediment flux (Q) with a variety of user inputs depending on the use case of the model run 
 - `utils` folder: Modules used in the calibration 
 - `mkdir_new_site.py`: Creates folders for filling with data. Allows you to follow a set organization structure for the repository. Makes all the scripts run nicely without need for constantly updating filepaths.
 - `pyscript.txt`: Bash script for running `calibration.py` in savio. 
 - `notebooks` folder: Jupyter notebooks used for cleaning and wrangling the raw remote sensing data, deriving flux measurements from sediment weights, and computing probability density functions by date for the wind measurements for each site (see the README in the notebooks folder for more information)<br>

To create an empty folder structure for a new site, activate the conda environment (see below) and run the following in the command line, replacing "site_name" with the new site: `python mkdir_new_site.py site_name`


## Activating the conda environment 
The calibration relies on a common set of python packages for data science, including numpy, pandas, scipy, and matplotlib. A yaml file is provided with an environment that will allow you to easily install the neccessary packages if you don't already have them installed. Setting this up requires that you have anaconda installed on your computer. 
<br><br>To create the environment from the file provided in this repository, simply run the following code in your terminal: 
<br>`conda env create -f environment.yml` 
<br><br>To activate the environment, named "wemo_calibration", run the following in your terminal: 
<br>`source activate wemo_calibration`


## Running the calibration script 
### Running the calibration from your local computer 
It's quite easy to run the calibration script locally. First, activate the conda environment following the instructions above. Then, from the root directory of the repository (i.e. the folder containing the calibration script), run the following line from terminal, replacing 1000 with your desired number of iterations: 
<br>`python calibration.py 1000`<br><br>
**This script requires that you input the number of iterations as a command line argument**. You can also add a second (optional) command line argument to add any descriptive text describing the calibration run, which will print to the output file. For example, say you want to make note of a change to the remote sensing indicators. You can adjust the command as such: 
<br>`python calibration.py 1000 'Using new remote sensing indicators for Moab and Mandan sites'`<br><br> 
The calibration script will save the results to the directory `data/results/YEAR-MON-DAY-HR-MIN/`, with the date being whenever you ran the script. This allows you to run the calibration several times in the same day. An output file is generated to describe the calibration settings; this file is saved as `calibration_info.txt` in the folder containing the results for that run.<br><br>
I've also added print statements throughout the script to indicate where you are in the calibration and also help with any debugging. The script will also print to the terminal an estimated time to completion, and a progress bar that updates with each iteration so that you can track the code's progress. 

### Running the calibration in the Berkeley Savio HPC cluser 
Savio is a [computing cluser](https://research-it.berkeley.edu/services-projects/high-performance-computing-savio) at UC Berkeley. Running the current calibration script with just the 3 NEAT sites for 1000 iterations each takes only 20 minutes on my personal laptop, so I haven't needed savio. However, you might need to use savio depending on the computational resources of your desktop/laptop computer and the calibration parameters you've set (i.e. running the calibration for many more sites or iterations). You can clone this repository and run the calibration in savio using a few simple steps: 

#### 1) Log into savio using your username 
 1) Run `ssh nicolekeeney@hpc.brc.berkeley.edu` in your terminal to log in to savio, replacing nicolekeeney with your savio username
#### 2) Clone the github repository to savio
 1) Follow the instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) to generate a personal access token attached to your GitHub account. This will be a collection of random letters/numbers that you'll need to copy. You'll also need to decide what permissions to give the key.
 2) Clone the repository by running the following in the directory where you want to store the repository (probably your user directory, i.e. scratch/users/nicolekeeney): <br>`git clone https://github.com/nicolejkeeney/wind_erosion_model_calibration` 
 3) After that, you'll be prompted to enter your GitHub username and a password. For the password, **input the personal access token, not your GitHub password.** 
 4) Cache your personal access token so you don't have to keep inputting it each time following the instructions. I've set it up using HTTPS, not SSH. Run `gh auth login` and input your username and personal access key.
#### 3) Create the conda environment 
 1) Migrate to the repository on savio (i.e. scratch/users/nicolekeeney/wind_erosion_model_calibration) 
 2) Load python in savio by running `module load python` 
 3) Create the conda environment by running `conda env create -f environment.yml`. **This step may take a while to complete** 
#### 4) Run the calibration script 
 1) Edit the text file, named `pyscript.txt ` in the repository, with your desired inputs. This will be at the bottom of the file, which reads: `python calibration.py x`, where x is the number of iterations desired; see the script for more information. 
 2) In terminal, from the repo directory, run `sbatch pyscript.txt`.
 3) If you want, check the job status using `squeue -u nicolekeeney`, replacing nicolekeeney with your savio username. 
