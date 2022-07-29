#!/usr/bin/env python
# coding: utf-8


"""calibration.py
    
    Minimize RMSEL to estimate best fit parameters for the WEMO model (Okin, 2008)
    Calibrated by following Li et al (2013), leaveout one site method 
    Remote sensing indicators are extracted using Zhou et al., 2020's algorithm (https://www.mdpi.com/2072-4292/13/2/283) 

    Author: Nicole Keeney 
    Date Created: 08/24/2021
    Modification History: 
        - Testing with new method: grabbing set of best fit parameters that minimized the MEAN of RMSEL for all sites; method referred to as "Nicole method" in code (10/13/2021)
        - Converted all loose code into functions with structure of a script with a main function (10/18/2021)
        - Changed log(A) range to -4 < log(A) < -1 and changed lower_gap_bins=[-1.50]. Changes noted in txt file in results dir (10/29/2021)
        - Test just NEAT sites (11/15/2021)
        - Use gap sizes as scale heights due to poor estimate of height from RS model; i.e. replace scale height computation with the gap bins (11/30/2021)
        - Change gap bins PDF such that the PDF doesn't need to sum to one; i.e. instead of summing 0.375, 0.75 to get bin for 0.75 and below, just use probability for 0.75 (11/30/2021). 
        - Compute ustar from windspeed, then compute PDF (instead of computing windspeed PDF, then converting to ustar in the model run) (01-28-2022)
        - Changed filename, set up scale height x or x/h as optional input (01-31-2022)
        - Returned back to computing windspeed PDF, then computing ustar from that (02-02-2022)
"""     

# Dependencies 
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from time import time 
from scipy.stats import linregress
from datetime import datetime

# Mute warnings 
import warnings
warnings.filterwarnings('ignore')

# Python modules 
from utils.progressBar import progressBar
from WEMO_model import WEMO_model
from utils.read_data_utils import read_all_site_data, read_windtower_data
from utils.model_utils import compute_shear_velocity, compute_scale_heights, compute_P_xh, compute_RMSEL, convert_time_elapsed, plot_Qpred_Qfield, df_to_csv



def main(num_iterations, notes=None):
    """ Run calibration and save results 
    See functions' docstrings for more information on the processes inputs and outputs and roles in the calibration process 
    
    """

    # ---------- Convert argparse strings to correct datatypes ----------
    try: 
        num_iterations = int(num_iterations)
    except: 
        raise ValueError("Input a valid integer value for the first argument, the number of iterations to run")
    
    # ---------- User defaults that can be easily modified ----------
    
    Li_method = False
    threshold_type = "POLARIS_perc_sand_updated" # Which set of u threshold values do you want to use? 
    #lower_gap_bins = [0.375, 0.75, 1.50] # Which gap bins to use? Assuming max_gap_bin will be added to the end 
    lower_gap_bins = [] # None 
    max_gap_bin = [3, 5, 7] # Maximim gap size bins to test 
    site_names_all = ["NEAT1","NEAT2","NEAT3","Moab","Holloman"] # Sites to use in calibration
    eq_to_test = ["mod_shao","owen","sorenson"] # Equations to test in calibration
    narr_data = True # Use wind tower measurements (set to False) or NARR data (set to True)? 
    scale_height_x = True # Use scale height = x (True) or scale height = x/h (False) ? 
    veg_subset = ["herbac_hgt"] # Vegetation types to use, used for computing scale height in scale height = x/h 

    # ---------- Directory to save results to ----------
    
    today = datetime.now()
    results_dir = "data/results/" + today.strftime("%Y-%b-%d-%H.%M") + "/"
  
    # Make directory
    if not os.path.isdir(results_dir): 
        os.makedirs(results_dir)
    
    # ---------- Create a text file with calibration info ----------

    textfile_name = results_dir+"calibration_info.txt"
    f = open(textfile_name, mode="w")
    f.write('Calibration Settings\n')
    f.write(f'Date: {today.strftime("%Y-%m-%d %H:%M")}\n')
    f.write(f'Number of iterations: {num_iterations}\n')
    f.write(f'Calibration method: {"Li method" if Li_method==True else "Nicole method"}\n')
    f.write("Sites used in calibration: "+", ".join(site_names_all)+"\n")
    f.write("Equations tested: "+", ".join(eq_to_test)+"\n")
    f.write(f'Wind data source: {"NARR 3 hourly 10m rasters" if narr_data==True else "tower data"}\n')
    f.write(f'Scale height computed as: scale height = {"x (where x = gap size)" if scale_height_x==True else "x/h (where x = gap size, h = vegetation height"}\n')
    if scale_height_x == False: 
        f.write("Vegetation type(s) used: "+", ".join(veg_subset)+"\n")
    if len(lower_gap_bins) == 0: 
        f.write("P(gap sizes): only using gap size > 2+\n")
    else: 
        f.write("P(gap sizes): Used gap size bins: "+", ".join([str(i) for i in lower_gap_bins+["2+"]])+"\n")
    f.write("Size of max gap bins: "+", ".join([str(i) for i in max_gap_bin]))
    if notes is not None: 
        f.write(f'\nAdditional notes: {notes}')
    f.close()
    
    # ---------- Function calls to run calibration & save results ----------

    print("SCRIPT STARTED")
    
    # Compute sets of best fit parameters
    # Saves 1 csv file: top_5_best_fit_params_by_leftout_date.csv 
    calibr_start_time = time()
    calibr_results_df = run_calibration(num_iterations=num_iterations, 
                                    site_names_all=site_names_all, 
                                    threshold_type=threshold_type, 
                                    lower_gap_bins=lower_gap_bins, 
                                    max_gap_bin=max_gap_bin, 
                                    veg_subset=veg_subset, 
                                    Li_method=Li_method, 
                                    results_dir=results_dir, 
                                    eq_to_test=eq_to_test,
                                    narr_data=narr_data, 
                                    scale_height_x=scale_height_x)

    f = open(textfile_name, mode="a") # Append calibration time to text file output 
    f.write(f'\nCalibration runtime: {convert_time_elapsed(time() - calibr_start_time)}')
    f.close()
    
    # Process dataframe output of previous function call to compute a single set best fit parameters 
    # Saves 2 csv files: best_fit_params_by_leftout_date.csv, best_fit_params_mean.csv
    A, C, U, z0, max_gap, eq_name, df_best, df_stats = process_calibration_results(calibr_results_df=calibr_results_df, site_names_all=site_names_all, results_dir=results_dir)

    # Test set of best fit parameters with WEMO model
    # Saves 2 csv files: rmsel_best_fit.csv, Q_pred_best_fit_all_sites_combined.csv
    # Saves png images with regression line/equation of Q_pred vs. Q_field for log and non-log for all sites combined and each site individually 
    run_model_best_fit(A=A, 
                       C=C, 
                       U=U, 
                       z0=z0, 
                       max_gap=max_gap, 
                       eq_name=eq_name, 
                       site_names_all=site_names_all, 
                       threshold_type=threshold_type, 
                       lower_gap_bins=lower_gap_bins, 
                       veg_subset=veg_subset, 
                       results_dir=results_dir, 
                       narr_data=narr_data, 
                       scale_height_x=scale_height_x)

    print("SCRIPT COMPLETE")





def run_model_best_fit(A, C, U, z0, max_gap, eq_name, site_names_all, threshold_type, lower_gap_bins, veg_subset, results_dir, narr_data, scale_height_x): 
    """ Run WEMO model with output of calibration
    Saves results as csv files 
    
    Args: 
        A (float): constant of variable units; output of process_calibration_results function
        C (float): e-folding distance for recovery of the shear stress in the lee of plants; output of process_calibration_results function
        U (float): shear velocity ratio in the immediate lee of plants; output of process_calibration_results function
        z0 (float): roughness length; output of process_calibration_results function
        max_gap (float): size for largest gap size bin; output of process_calibration_results function
        eq_name (str): name of Q equation; output of process_calibration_results function
        site_names_all (list of str): Site names to use. Must correspond to site names in data/site_names folder 
        threshold_type (str): threshold values to use. Must correspond to a column in data/site_data/u_threshold_by_site.csv
        lower_gap_bins (list of float): gap size bins to use for all but max gap bin; i.e. [0.375, 0.75, 1.5]. Max gap will be appended to end of list 
        veg_subset (list of str): subset of vegetation type/s to use. Must correspond to vegetation types in data/site_data/RS_indicators/site_name_veg_hgts.csv
        results_dir (str): path to save file to (this should just be the folder path, not including the filename of the csv (i.e. data/, NOT data/filename.csv. The filename is set in the code))
        narr_data (bool): Use NARR rasters for wind data (set to True) or wind tower data (set to False)
        scale_height_x (bool): Use scale height = x (True) or scale height = x/h (False) ? 
    
    Returns: 
        None
    """
    
    # Initialize figures directory 
    figs_dir = results_dir+"figs/"
    if not os.path.isdir(figs_dir): 
        os.makedirs(figs_dir)

    # Run WEMO model and compute RMSEL for each site using the mean values of the parameters computed in the calibration
    print("\nRunning the WEMO model for all sites using the mean best fit parameters from the calibration...")
    df_results = pd.DataFrame(index=["RMSEL","slope","intercept","Er","corr_coefficient","num_values"])
    Q_all_df = []
    for site_name in site_names_all: 

        # Read in threshold data 
        u_threshold_df = pd.read_csv("data/site_data/u_threshold_by_site.csv", index_col="Sitename")
        u_threshold_df = u_threshold_df[threshold_type] # Grab u threshold by user inputted threshold type 
        u_threshold = u_threshold_df.loc[site_name]

        # Read in data 
        Q_field, veg_hgts, gap_sizes, tot_folia, wind_pdf = read_all_site_data(site_name=site_name, data_dir="data/site_data/", narr_data=narr_data)

        # Compute ustar PDF 
        ustar = compute_shear_velocity(u_z=wind_pdf.columns.astype(float).values, # Windspeed in m/s
                                       z0=z0, # Roughness length 
                                       zheight=10) # Height of measurements 
        ustar_pdf = wind_pdf.copy()
        ustar_pdf.columns = ustar
        
        # Compute P(x/h)
        P_xh_df = gap_sizes[[str(num) for num in lower_gap_bins +["2+"]]].copy() 

        # Compute scale height
        if scale_height_x == True: # Set up scale height = x      
            scl_hgt = pd.DataFrame(data=np.full(P_xh_df.shape, [float(num) for num in lower_gap_bins +[max_gap]]), 
                                   columns=[str(num) for num in lower_gap_bins +[max_gap]], 
                                   index=P_xh_df.index) 
        if scale_height_x == False: # Compute scale height = x/h
            scl_hgt = compute_scale_heights(veg_hgts[veg_subset], # Vegetation heights cut to the subset of vegetation types to use 
                            gap_sizes_list=lower_gap_bins +[max_gap]) # Gap size bins to use 

        # Run WEMO model for that site for the set of random parameters 
        Q_pred = WEMO_model(scl_hgt=scl_hgt, # Scale heights 
                            Pd_xh=P_xh_df, # Gap sizes pd.df 
                            ustar_pdf=ustar_pdf, # Wind probability density function
                            Fg=tot_folia, # Total fractional foliage (0-1) per date 
                            A=A, C=C, U=U, z0=z0, # Input parameters  
                            u_threshold=u_threshold, # Threshold friction velocity
                            q_pred_eq=eq_name) # Equation to use 

        # Get numpy arrays from dataframes 
        Qpred_np = Q_pred["Q_pred"].values
        Qfield_np = Q_field["Q_field"].values
        Qpred_np_no_nan = Qpred_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]
        Qfield_np_no_nan = Qfield_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]
        num_values_used = len(Qfield_np_no_nan)

        # Compute slope and intercept of log10(Q pred) vs. log10(Q field)
        slope, intercept, rvalue, pvalue, stderr = linregress(x=np.log10(Qfield_np_no_nan), y=np.log10(Qpred_np_no_nan))

        # Compute RMSEL
        rmsel_i = compute_RMSEL(Qpred_np=Qpred_np, Qfield_np=Qfield_np)
        Er = 10**rmsel_i - 1

        # Add to dataframe 
        df_results[site_name] = [rmsel_i, slope, intercept, Er, rvalue, num_values_used]

        # Append to list 
        Q_df = pd.concat([Q_pred, Q_field], axis=1)
        Q_df["site_name"] = site_name
        Q_all_df.append(Q_df)

        fig_by_site = plot_Qpred_Qfield(Qpred_np=Qpred_np, Qfield_np=Qfield_np, 
                                        title="Site: "+site_name, 
                                        figs_path=figs_dir+site_name+".png")

    # Combine dataframes with results 
    Q_all_df = pd.concat(Q_all_df)
    Q_all_df = Q_all_df.dropna()
    fig_all = plot_Qpred_Qfield(Qpred_np=Q_all_df["Q_pred"].values, Qfield_np=Q_all_df["Q_field"].values, 
                                title="Sites: "+", ".join(site_names_all), 
                                figs_path=figs_dir+"all_sites.png")

    # Add descriptive info and print results
    df_results = df_results.T
    df_results.index.name = "Site name"
    print(df_results[["RMSEL","Er","num_values"]])

    # Save results 
    df_to_csv(df=df_results, 
              directory_path=results_dir, 
              filename="rmsel_best_fit.csv")
              
    df_to_csv(df=Q_all_df, 
              directory_path=results_dir, 
              filename="Q_pred_best_fit_all_sites_combined.csv")
    
    return None





def process_calibration_results(calibr_results_df, site_names_all, results_dir): 
    """ Use results dataframe from the calibration to compute the single set of best fit parameters 
    To do this, we grab the single set of best fit parameters by leftout site, giving num_sites number of parameters. 
    A set of parameters is defined as values for log(A), C, U, log(z0), max_gap, and eq_name 
    Then, from the set of sets of parameters, we compute a mean for each log(A), C, U, log(z0), and a mode for max_gap and eq_name, resulting in a single value for each 
    Dataframes are saved as csv files in results_dir

    Args: 
        calibr_results_df (pd.DataFrame): dataframe outputted by calibration function containing sets of parameters for each left out site and information on the RMSEl for each set 
        site_names_all (list of str): Site names to use. Must correspond to site names in data/site_names folder 
        results_dir (str): path to save file to (this should just be the folder path, not including the filename of the csv (i.e. data/, NOT data/filename.csv. The filename is set in the code))
    
    Returns: 
        A (float): constant of variable units; mean from df_stats  
        C (float): e-folding distance for recovery of the shear stress in the lee of plants; mean from df_stats  
        U (float): shear velocity ratio in the immediate lee of plants; mean from df_stats  
        z0 (float): roughness length; mean from df_stats  
        max_gap (float): size for largest gap size bin; mode from df_stats   
        eq_name (str): name of Q equation; mode from df_stats 
        df_best (pd.DataFrame): single set of best fit parameters for each leftout site
        df_stats (pd.DataFrame): mean of A,C,U,and z0, mode of max_gap and eq_name
    
    """
    
    # Compute df_best 
    print("Grabbing the single best fit parameters by leftout site, saving as pandas DataFrame...")
    df_best = pd.DataFrame()
    for site in site_names_all:
        best_fit_by_leftout_site = calibr_results_df[calibr_results_df["leftout_site"]==site].iloc[0,:] # Best fitting parameters for that one leftout site. I.e. the first column in best_fit)df
        df_best[site] = best_fit_by_leftout_site.drop(labels="leftout_site")
    df_best = df_best.T
    df_best.index.name = "leftout_site"
    df_best["A(x10^-3)"] = (10**df_best["log(A)"])*(10**3)
    df_best["z0"] = 10**df_best["log(z0)"]
    df_best = df_best[["A(x10^-3)","C","U","z0","max_gap","Qpred_equation","log(A)","log(z0)"]]
    print(df_best[["A(x10^-3)","C","U","z0","max_gap"]])

    # Get the mean and standard deviation of the columns 
    df_stats = pd.DataFrame([df_best.mean(axis=0), df_best.std(axis=0)],index=["mean","std"])
    df_stats = df_stats[["log(A)","C","U","log(z0)","max_gap"]]

    # Grab data from the dataframe 
    eq_name = df_best["Qpred_equation"].mode().values[0] # Get most common value
    max_gap = df_best["max_gap"].mode().values[0] # Get most common value 
    print("\nMost common (mode) values used for: \nmaximum gap size bin (meters) = {0}\nQ flux equation type = {1}".format(max_gap, eq_name))
    logA, C, U, logz0 = df_stats.loc["mean"][["log(A)","C","U","log(z0)"]].values # Use mean values for the parameters
    A = 10**logA
    z0 = 10**logz0
    print("\nMean best fit values: \nconstant A(x 10^-3) (variable units) = {:.4f}\ne-folding distance (C, dimensionless) = {:.2f}\nshear velocity ratio (U, dimensionless) = {:.3f}\nroughness length (z0, meters) = {:.2f}\nmaximum gap size (meters) = {:.2f}\n".format(A*10**3,C,U,z0,max_gap))
    
    # Add mode to df_stats 
    df_stats["equation type (mode)"] = [eq_name, np.nan]
    df_stats["max gap (mode)"] = [max_gap, np.nan]
    
    # Save results 
    df_to_csv(df=df_best, 
              directory_path=results_dir, 
              filename="best_fit_params_by_leftout_site.csv")
    # Save results 
    best_fit_params_mean = df_stats.T
    best_fit_params_mean.index.name = "parameter"
    df_to_csv(df=best_fit_params_mean, 
              directory_path=results_dir, 
              filename="best_fit_params_mean.csv")
    
    return A, C, U, z0, max_gap, eq_name, df_best, df_stats 





def run_calibration(num_iterations, site_names_all, threshold_type, lower_gap_bins, max_gap_bin, veg_subset, Li_method, results_dir, eq_to_test, narr_data, scale_height_x):
    """ Run WEMO model calibration 
    Print statements help with debugging 
    The Li method represents grabbing the set of parameters associated with the lowest RMSEL after running num_iterations for each leftout site; this normally results in grabbing the set of parameters for NEAT3
    The Nicole method (use by setting Li_method=False) gravs the set of parameters associated with the lowest mean RMSEL of all sites for each leftout site, in an attempt to the bias toward NEAT3 
    
    Args: 
        num_iterations (int): Number of iterations to run calibration for. This will be the num sites * num iterations; i.e. set num iterations to 100 to get 100*num sites total iterations 
        site_names_all (list of str): Site names to use. Must correspond to site names in data/site_names folder 
        threshold_type (str): threshold values to use. Must correspond to a column in data/site_data/u_threshold_by_site.csv
        lower_gap_bins (list of float): gap size bins to use for all but max gap bin; i.e. [0.375, 0.75, 1.5]. Max gap will be appended to end of list 
        max_gap_bin (list of float): bin/s to test for final gap bin. This will be iterated through as the final gap bin in the form [lower_gap_bins, max_gap_bin_i] for each value in max_gap_bin
        veg_subset (list of str): subset of vegetation type/s to use. Must correspond to vegetation types in data/site_data/RS_indicators/site_name_veg_hgts.csv
        Li_method (bool): Use Li method (True) or Nicole's method (False)? 
        results_dir (str): path to save file to (this should just be the folder path, not including the filename of the csv (i.e. data/, NOT data/filename.csv. The filename is set in the code))
        eq_to_test (list): Equations to test calibration for 
        narr_data (bool): Use NARR rasters for wind data (set to True) or wind tower data (set to False)
        scale_height_x (bool): Use scale height = x (True) or scale height = x/h (False) ? 

    Returns:
        final_df (pd.DataFrame): Set of 5 best fit parameters for each leftout site, giving a total of 5*num_sites rows in the dataframe, along with the RMSEL, slope, intercept, and r of the logs Qfield vs. Qpred. For the Nicole method, mean RMSEL is also included as a column 
        
    """

    start_time = time()
    print("Starting calibration")
    print("\nRunning calibration for " + str(num_iterations) + " iterations") 

    # Read in threshold data 
    u_threshold_df = pd.read_csv("data/site_data/u_threshold_by_site.csv", index_col="Sitename")
    u_threshold_df = u_threshold_df[threshold_type] # Grab u threshold by user inputted threshold type 
    u_threshold_df = u_threshold_df.loc[site_names_all]
    
    
    # Save threshold dataframe as csv 
    df_to_csv(df=u_threshold_df, 
              directory_path=results_dir, 
              filename="u_threshold_by_site.csv")

    # Print info for user
    print("\nUsing Li et al method? "+str(Li_method))
    print("\nSites used in calibration: "+", ".join(site_names_all))
    print("\nEquations tested: "+", ".join(eq_to_test))
    if scale_height_x == True: 
        print("\nUsing scale height = x")
    elif scale_height_x == False: 
        print("\nUsing scale height = x/h, where h is the vegetation height")
        print("\nVegetation types used for h: "+", ".join(veg_subset))
    if narr_data == True: 
        print("\nUsing NARR rasters for ustar PDF computation")
    elif narr_data == False: 
        print("\nUsing wind tower data for ustar PDF computation")
    if len(lower_gap_bins) == 0: 
        print("\nOnly using gap size bins > 2+ ")
    else: 
        print("\nLower gap size bins to use: "+", ".join([str(i) for i in lower_gap_bins]))
        print("\nMax gap size bins to test: "+", ".join([str(i) for i in max_gap_bin]))
    print("\nThreshold friction velocity by site: ")
    for sitename, uthresh in u_threshold_df.iteritems():
        print(f"{sitename}: {uthresh}")

    # Loop through each site! 
    j = 0 # For progress bar
    best_fit_params_for_leftout_site = [] # This is where you'll store the best fit paramater values for each left out site

    for site_to_leave_out in site_names_all: 

        # Get a list of all site names EXCEPT the one site we are leaving out 
        sites_to_run_analysis_for = [site for site in site_names_all if site!=site_to_leave_out]

        # Set up loop 
        i = 0
        params_list, rmsel_list, gap_list, site_name_list, eq_list, slope_list, intercept_list, r_list = [],[],[],[],[],[],[],[]
        while i < num_iterations:

            i+=1

            # Generate random values for the parameters 
            log_A = np.random.uniform(low=-4, high=-1)
            C = np.random.uniform(low=4.8, high=9)
            U = np.random.uniform(low=0, high=0.4)
            log_z0 = np.random.uniform(low=-2, high=-0.3)

            A = 10**(log_A)
            z0 = 10**(log_z0)
            
            for site_name in sites_to_run_analysis_for: 
                
                # Get threshold shear velocity
                u_threshold = u_threshold_df.loc[site_name]

                # Read in data 
                Q_field, veg_hgts, gap_sizes, tot_folia, wind_pdf = read_all_site_data(site_name=site_name, data_dir="data/site_data/", narr_data=narr_data)

                # Compute ustar PDF 
                ustar = compute_shear_velocity(u_z=wind_pdf.columns.astype(float).values, # Windspeed in m/s
                                               z0=z0, # Roughness length 
                                               zheight=10) # Height of measurements 
                ustar_pdf = wind_pdf.copy()
                ustar_pdf.columns = ustar
        
                # Compute P(x/h)
                P_xh_df = gap_sizes[[str(num) for num in lower_gap_bins +["2+"]]].copy() 
                        
                
                for max_gap in max_gap_bin: # Loop through gap sizes
                 
                    # Compute scale height
                    if scale_height_x == True: # Set up scale height = x      
                         scl_hgt = pd.DataFrame(data=np.full(P_xh_df.shape, [float(num) for num in lower_gap_bins +[max_gap]]), 
                                                columns=[str(num) for num in lower_gap_bins +[max_gap]], 
                                                index=P_xh_df.index) 
                    if scale_height_x == False: # Compute scale height = x/h
                        scl_hgt = compute_scale_heights(veg_hgts[veg_subset], # Vegetation heights cut to the subset of vegetation types to use 
                                                        gap_sizes_list=lower_gap_bins +[max_gap]) # Gap size bins to use 
                    
                    for eq_name in eq_to_test: # Loop through equations for Q(t,pred). See WEMO_model.py for more information. 
                        
                        # Run WEMO model for that NEAT site for the set of random parameters 
                        Q_pred = WEMO_model(scl_hgt=scl_hgt, # Scale heights 
                                            Pd_xh=P_xh_df, # Gap sizes pd.df 
                                            ustar_pdf=ustar_pdf, # Wind probability density function
                                            Fg=tot_folia, # Total fractional foliage (0-1) per date 
                                            A=A, C=C, U=U, z0=z0, # Input parameters  
                                            u_threshold=u_threshold, # Threshold friction velocity
                                            q_pred_eq=eq_name) # Equation to use 
                 
                        # Get numpy arrays from dataframes 
                        Qpred_np = Q_pred["Q_pred"].values
                        Qfield_np = Q_field["Q_field"].values
                        Qpred_np_no_nan = Qpred_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]
                        Qfield_np_no_nan = Qfield_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]

                        # Compute slope and intercept of log10(Q pred) vs. log10(Q field)
                        slope, intercept, rvalue, pvalue, stderr = linregress(x=Qfield_np_no_nan, y=Qpred_np_no_nan)    

                        # Compute RMSEL
                        rmsel_i = compute_RMSEL(Qpred_np=Qpred_np, Qfield_np=Qfield_np)

                        # Save results of loop
                        params_list.append([log_A, C, U, log_z0]) # Value of each parameter
                        rmsel_list.append(rmsel_i) # A list of 3 values: RMSEL for each of the 3 NEAT sites
                        gap_list.append(max_gap) # Size of final gap bin  
                        site_name_list.append(site_name) # Name of site 
                        eq_list.append(eq_name) # Name of equation used to compute Q(t,pred)
                        slope_list.append(slope)
                        intercept_list.append(intercept)
                        r_list.append(rvalue)

            # Print estimated runtime
            if j == 0: 
                time_for_one_loop = time() - start_time # Elapsed time for one loop
                time_to_completion = time_for_one_loop*len(site_names_all)*num_iterations # Time for all loops
                print("\nEstimated time to completion: "+convert_time_elapsed(time_to_completion))

            # Output progress bar 
            progressBar(j, len(site_names_all)*num_iterations) 
            j+=1

        # Compile results
        num_sites = len(sites_to_run_analysis_for)
        params_df = pd.DataFrame(params_list, columns=["log(A)","C","U","log(z0)"])
        rmsel_gap_df = pd.DataFrame([gap_list,rmsel_list,site_name_list,eq_list, slope_list,intercept_list,r_list], index=["max_gap","RMSEL","site_name","Qpred_equation","slope","intercept","r"]).T
        results_df = pd.merge(params_df, rmsel_gap_df, left_index=True, right_index=True)
        results_df["leftout_site"] = site_to_leave_out

        # Get best fit parameters 
        num_results_per_leftout_site = 5 # How many best fit results do you want? Set to 5 to get the top 5 sets of parameters for each leftout site 
        if Li_method==True: 
            results_sorted = results_df.sort_values(axis=0, by=["RMSEL"], ascending=True).reset_index(drop=True)
            best_fit_df = results_sorted.iloc[0:num_results_per_leftout_site,:] # Top 5 best fit 

        else: 
            df = results_df.groupby(["log(A)","C","U","log(z0)","max_gap","Qpred_equation","site_name"]).min()
            df["mean_RMSEL"] = df["RMSEL"].sum(level=["log(A)","C","U","log(z0)","max_gap","Qpred_equation"]) / num_sites
            df = df.sort_values(axis=0, by=["mean_RMSEL","RMSEL"], ascending=True)
            top_few_best_all_sites = df.iloc[0:num_sites*num_results_per_leftout_site]
            best_fit_df = top_few_best_all_sites.reset_index(drop=False)[::num_sites].reset_index(drop=True)

        best_fit_params_for_leftout_site.append(best_fit_df)
        
    # Print total execution time 
    print("\nCALIBRATION COMPLETE!\nExecution time: "+convert_time_elapsed(time() - start_time))

    # Combine results from each site 
    # 5 best fit parameters, by leftout date
    final_df = pd.concat(best_fit_params_for_leftout_site)
    final_df = final_df.convert_dtypes()

    # If final df is totally empty, exit script
    if len(final_df) == 0: 
        print("Warning: no results computed... try running for more iterations. Outputting empty dataframe and terminating script.")
        exit()
    
    # Save results 
    df_to_csv(df=final_df, 
              directory_path=results_dir, 
              filename="top_"+str(num_results_per_leftout_site)+"_best_fit_params_by_leftout_date.csv")

    return final_df 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calibrate WEMO model to compute set of best fit parameter values following leavout-one-out calibration method detailed in Li et al (2013)')
    parser.add_argument('num_iterations', help='Number of iterations to perform')
    parser.add_argument('notes', nargs='?', const='arg_was_not_given', help='Extra notes to add to text file about calibration details')
    args = parser.parse_args()
    main(num_iterations=args.num_iterations, notes=args.notes)

