""" utils.py 

    Non-categorized helper functions for WEMO model calibration 
    
    Author: Nicole Keeney 
    Date Created: 06-22-2021
    Modification History:
        - Added compute RMSEL function (06-23-2021)
        - Added convert_time_elapsed function, credit to Geeks for Geeks (09-29-2021)
    
"""

import os
import numpy as np 
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt


def compute_shear_velocity(u_z, z0, zheight=10): 
    """Compute shear velocity at surface from windspeed at a given height. 
    
    Args: 
        u_z (list or np.array): speed of wind in m/s at height zheight 
        z0 (float): roughness length 
        zheight (float/int): height at which wind measurement was taken in meters (default to 10)
    
    Returns: 
        ustar (float): shear velocity in m/s
    
    """
    K = 0.4       # Von Kármán constant
    ustar = (u_z * K)/(np.log(zheight/z0)) # Compute ustar
    return ustar


def df_to_csv(df, directory_path, filename): 
    if not os.path.isdir(directory_path): 
        os.makedirs(directory_path)
        #print("\nCreated directory "+directory_path)
    total_filepath = directory_path+filename
    df.to_csv(total_filepath) 
    print("\nSaved results to "+total_filepath)


def compute_RMSEL(Qpred_np, Qfield_np): 
    """ Compute log of the root mean square error using the equation from Li et al (2013) for predicted and measured horizontal aeolian flux (Q)
    
    Args: 
        Qpred_np (np.array): array of predicted flux by date 
        Qfield_np (np.array): array of field measured flux by date 
        
    Returns: 
        RMSEL (float): log of the root mean square error
    
    """
    # Remove nans 
    Qpred_np_no_nan = Qpred_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]
    Qfield_np_no_nan = Qfield_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]
    
    # Compute RMSEL 
    n = len(Qpred_np_no_nan)
    n_sum = sum((np.log10(Qpred_np_no_nan) - np.log10(Qfield_np_no_nan))**2)
    RMSEL = np.sqrt((1/n)*n_sum)
    return RMSEL


def compute_scale_heights(veg_hgts, gap_sizes_list=[0.375, 0.75, 1.50, 2.0]):
    """ Compute scale heights by gap size for a subset of vegetation 
    
    Args: 
        veg_hgts (pd.DataFrame): vegetation heights by date for a set of vegetation types
        gap_sizes_list (list of four floats, optional): sizes of gaps in cm 
        
    Returns: 
        scl_hgt (pd.DataFrame): scale height by gap size for each date in veg_hgts 
    
    """
    
    mean_val_ar = veg_hgts.mean(axis=1).values # Compute mean value of the vegetation heights 
    
    scl_hgt_list = []
    for gap in gap_sizes_list: 
        scl_hgt_gap_i = gap / mean_val_ar # Divide each gap bin by the mean vegetation height at each date 
        scl_hgt_list.append(scl_hgt_gap_i)                 

    scl_hgt = pd.DataFrame(data=np.array(scl_hgt_list).T, # Array of scale height by gap size by date 
                           columns=gap_sizes_list, # Columns are the gap size bins 
                           index=veg_hgts.index) # Index is the dates 
    scl_hgt.columns.name = "Gap size"
    return scl_hgt


def compute_P_xh(gap_sizes_df, gap_bins_list):  
    """ Compute Pd(x/h) using the gap sizes probability density function 
    If gap_bins_list = [0.375, 0.75, 1.5, x], where x is any float, P_d_xh = gap_sizes_df, with no changes made 
    If gap_bins_list = [1.5, x], the function sums the rows for 0.375, 0.75, and 1.5 to create one gap bin for 1.5, returning a dataframe with two columns [1.5, x]
    Pd(x/h) is defined by Li et al (2013) as the probability that any point in the landscape is a certain distance from the nearest upwind plant expressed in units of height of that plant.
    
    Args: 
        gap_sizes_df (pd.DataFrame): gap sizes probability density function 
        gap_bins_list (list of floats): list of gap size bins 
        
    Returns: 
        P_d_xh (pd.DataFrame): gap size probability density function

    """
    P_d_xh = gap_sizes_df.copy()
    if gap_bins_list[:-1] == [1.5]: 
        P_d_xh["1.5"] = P_d_xh[["0.375","0.75","1.5"]].sum(axis=1)
        P_d_xh = P_d_xh.drop(columns=["0.375","0.75"])
    elif gap_bins_list[:-1] == [0.75, 1.5]: 
        P_d_xh["0.75"] = P_d_xh[["0.375","0.75"]].sum(axis=1)
        P_d_xh = P_d_xh.drop(columns=["0.375"])
    elif gap_bins_list[:-1] == [0.375, 1.5]:
        P_d_xh["1.5"] = P_d_xh[["0.75","1.5"]].sum(axis=1)
        P_d_xh = P_d_xh.drop(columns=["0.75"])
    elif gap_bins_list[:-1] == [[1.5],[0.75, 1.5],[0.375, 1.5],[0.375, 0.75, 1.5]]: 
        pass
    elif gap_bins_list[:-1] != [0.375, 0.75, 1.5]: 
        raise ValueError("Your input for gap bins list is not valid.\nYour input: "+str(gap_bins_list))
    return(P_d_xh)


def plot_Qpred_Qfield(Qfield_np, Qpred_np, figs_path=None, title=None, figsize=(12,4), dpi=200): 
    """ Generate plots of predicted flux (y) vs. measured flux (x)
    Performs a linear regression to compute slope and intercept to plot a best fit line 
    
    Args: 
        Qfield_np (np.array): array of field measured flux by date 
        Qpred_np (np.array): array of predicted flux by date 
        figs_path (str, optional): path to save figure to; must contain ".png" at end (default to None-- do not save figure)
        title (str, optional): title for plot (default to None)
        figsize (tuple, optional): size of figure (default to 12,4)
        dpi (int, optional): number of pixels composing saved figure (default to 200)
        
    Returns: 
        fig (matplotlib figure)
    
    """
    # Set up plot with 2 axes 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    # Remove nans 
    Qpred_np_no_nan = Qpred_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]
    Qfield_np_no_nan = Qfield_np[~(np.isnan(Qpred_np)) & ~(np.isnan(Qfield_np))]

    # Plot log
    x = np.log10(Qfield_np_no_nan)
    y = np.log10(Qpred_np_no_nan)
    slope, intercept, rvalue, pvalue, stderr = linregress(x=x, y=y)
    ax1 = axes[0]
    ax1.scatter(x, y)
    y_pred = slope*np.sort(x) + intercept
    ax1.plot(np.sort(x), y_pred, color="black",linestyle='dashed',label="y={0:.2f}x+{1:.2f}".format(slope,intercept))
    ax1.set_title("log(Q field) vs. log(Q pred)")
    ax1.set_ylabel("log10(Q predicted)")
    ax1.set_xlabel("log10(Q field)")
    ax1.legend()

    # Plot Q_pred vs Q_act
    x = Qfield_np_no_nan
    y = Qpred_np_no_nan
    slope, intercept, rvalue, pvalue, stderr = linregress(x=x, y=y)
    ax2 = axes[1]
    ax2.scatter(x, y, color='green',marker='s')
    y_pred = slope*np.sort(x) + intercept
    ax2.plot(np.sort(x), y_pred, color="black",linestyle='dashed',label="y={0:.2f}x+{1:.2f}".format(slope,intercept))
    ax2.set_title("Q field vs. Q pred")
    ax2.set_ylabel("Q predicted (g/m/day)")
    ax2.set_xlabel("Q field (g/m/day)")
    ax2.legend(loc="upper left")
    
    # Give title to figure 
    fig.suptitle(title, y=1.04, fontsize=15)
    
    # Save fig 
    if (figs_path is not None): 
        if figs_path.endswith(".png") == False: 
            figs_path+=".png"
        plt.savefig(figs_path, facecolor="white", dpi=dpi, bbox_inches='tight')
    
    return fig 


def convert_time_elapsed(seconds):
    """ Convert seconds to hr:min:sec format for better readability
    Function credit to geeksforgeeks.org
    https://www.geeksforgeeks.org/python-program-to-convert-seconds-into-hours-minutes-and-seconds/
    
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60      
    return "%d:%02d:%02d" % (hour, minutes, seconds)