"""mkdir_new_site.py

    Creates folders for filling with data. 
    Allows you to follow a set organization structure for the repository. Makes all the scripts run nicely without need for constantly updating filepaths. 

    Author: Nicole Keeney 
    Date Created: 08-21-2021 
    Modification History: 
        - Changed format to allow for running with command line arguments (02-09-2022)

"""

import os 
import time
import argparse


def main(site_name): 

    print("Creating folder structure for the following site name: "+site_name)

    # Change directories into the main repository
    start = time.time()
    repo_name = "WEMO_calibration" # Name of github repository. Home directory for code 
    while os.path.basename(os.getcwd()) != repo_name:    # Check if basepath is repository name 
        if time.time() - start > 5: # See if the loop has been running for a while... this usually indicates a user error
            raise ValueError("This loop has been running for a while... you should manually inspect code. \nMaybe the repository name you inputted is wrong, or you're in a directory that doesn't make sense. \nCurrent directory: "+os.getcwd()+"\nInput repository name: "+repo_name)
        os.chdir("..")

    create_and_change_dir(dir_name="data", change_dir=True) # Move into data directory 
    create_and_change_dir(dir_name="site_data", change_dir=True) # Move into site directory 
    create_and_change_dir(dir_name=site_name, change_dir=True) # Create site name folder 
    create_and_change_dir(dir_name="field_Q", change_dir=False) # Directory for storing horizontal sediment flux data measured in the field 
    create_and_change_dir(dir_name="wind", change_dir=False) # Directory for storing wind probability density function 
    create_and_change_dir(dir_name="RS_indicators", change_dir=False) # Directory for storing remote sensing derived info

    print("COMPLETE")


def create_and_change_dir(dir_name, change_dir=True): 
    """ Change into directory dir_name. If directory does not exist, create it first! 
    
    Args: 
        dir_name (str): Name of directory 
        change_dir (bool, optional): Change directories into dir_name? (default to True)
    
    """
    
    if dir_name not in os.listdir(os.getcwd()): # Create directory if it doesn't already exist 
        os.mkdir(dir_name)
        print("Created directory "+dir_name)
    if change_dir==True: 
        os.chdir(dir_name)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make set of empty directories for a new site')
    parser.add_argument('site_name', help='Name of site')
    args = parser.parse_args()
    main(site_name=args.site_name)