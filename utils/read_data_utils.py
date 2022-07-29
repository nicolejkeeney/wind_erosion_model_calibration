"""read_data_utils.py 

   Helper functions for reading in data
   
   Author: Nicole Keeney 
   Date Created: 06-17-2021 
   Modification History:
       - Changed filename, added read_neat_data function (06-22-2021)
       - Added read_all_site_data function (09-29-2021)

"""

import pandas as pd
import os



def read_datetime_csv(path, date_col="Date", strftime=None):
    """ Read in csv file with Date as index. Convert date index to pd.Datetime type. 
    
    Args: 
        path (str): path to csv 
        date_col (str, optional): name of date column (default to "Date")
        strftime (str, optional): datetime format of date column (i.e if dates are set to the format 20150820, set strftime="%Y%m%d")
    Returns: 
        df (pd.DataFrame)
    
    """
    df = pd.read_csv(path, index_col=date_col) # Read in csv 
    df.index = pd.to_datetime(df.index, format=strftime) # Convert index to pd.Datetime type 
    df = df.sort_index() # Sort by datetime index 
    return df 


def read_windtower_data(site_name, data_dir="data/site_data/"): 
    """ Read in wind tower data for a site of interest 
    Convert knots to m/s
    
    Args: 
        site_name (str): site to grab data for 
        data_dir (str): repository containing wind data 
    
    Returns: 
        wind (pd.Series): tower data by datetime 
    
    """
    wind_path = data_dir+site_name+"/wind/"+site_name+"_wind_raw.csv" # Wind file
    df_wind = read_datetime_csv(wind_path, date_col="valid") # Read in wind data 
    df_wind.index.name = "date"
    df_wind.sknt = pd.to_numeric(df_wind.sknt, errors = 'coerce') # Convert the missing data to NaN
    df_wind["windspeed (m/s)"] = df_wind.sknt * 0.51 # Convert knots --> m/s 
    wind = df_wind["windspeed (m/s)"]
    return wind


def read_neat_data(dataDir, neat_num): 
    """ Read in data from NEAT folder 
    
    Args:     
        - dataDir (str): directory containing the folders neat1, neat2, neat3 (must contain "/" after the folder path)
        - neat_num (str): number corresponding to NEAT site to read in
    
    Returns: tuple 
        - Q_df (pd.DataFrame): Q data at the NEAT site
        - veg_df (pd.DataFrame): vegetation heights at the NEAT site
        - gap_df (pd.DataFrame): gap sizes at the NEAT site 
        - folia_df (pd.DataFrame): total foliage at the NEAT site
    
    """
    # Define directory and filepaths
    NEAT_folderPath = dataDir+"/neat"+neat_num+"/" # Path to neat directory    
    Q_path = NEAT_folderPath + "Q_NEAT" + neat_num + ".csv"
    veg_path = NEAT_folderPath + "canopy_height_ecotype" + neat_num + ".csv"
    gap_path = NEAT_folderPath + "gap_sizes" + neat_num + ".csv"
    tot_fol_path = NEAT_folderPath + "total_foliage" + neat_num + ".csv"
    
    # Check that paths are valid 
    if not os.path.isdir(NEAT_folderPath): 
        raise ValueError("The argument dataDir is not a valid path: " + dataDir)
    for path in [Q_path, veg_path, gap_path]:
        if not os.path.isfile(path): 
            raise ValueError("The path is not valid: " + path)
    
    # Read in data as pd.DataFrame objects
    Q_df = read_datetime_csv(Q_path)
    veg_df = read_datetime_csv(veg_path)
    gap_df = read_datetime_csv(gap_path)
    folia_df = read_datetime_csv(tot_fol_path)
    
    return Q_df, veg_df, gap_df, folia_df



def read_all_site_data(site_name, data_dir="data/site_data/", narr_data=False): 
    """ Read in all relevant data for the WEMO model for a site name of interest 
    All sites follow the name filename conventions, allowing for this to be wrapped into a function and used for all the different sites
    
    Args: 
        site_name (str): name of site to grab data for 
        data_dir (str, optional): path to main data directory (default to "data/site_data/")
        narr_data (bool): Use NARR rasters for wind data (set to True) or wind tower data (set to False) (default to False)
    
    Returns: tuple
        Q_field (pd.DataFrame): field measurements of sediment flux by date
        veg_hgts (pd.DataFrame): vegetation height measurements by date for different vegetation types
        gap_sizes (pd.DataFrame): gap sizes by date 
        tot_foliar (pd.DataFrame): total foliage cover by date
        wind_pdf (pd.DataFrame): windspeed probability density function by date 
    
    """
    
    Q_field = read_datetime_csv(data_dir+site_name+"/field_Q/"+site_name+"_fieldQ.csv")
    veg_hgts = read_datetime_csv(data_dir+site_name+"/RS_indicators/"+site_name+"_veg_hgts.csv")
    gap_sizes = read_datetime_csv(data_dir+site_name+"/RS_indicators/"+site_name+"_gapsize_norm.csv")
    tot_folia = read_datetime_csv(data_dir+site_name+"/RS_indicators/"+site_name+"_perc_fol_cover.csv")
    
    if narr_data == True: 
        wind_pdf = read_datetime_csv(data_dir+site_name+"/wind/"+site_name+"_NARR_wind_pdf.csv")
    else: 
        wind_pdf = read_datetime_csv(data_dir+site_name+"/wind/"+site_name+"_wind_pdf.csv")
        
    if "NEAT" in site_name: 
           # Remove last date until Abi can get you the full RS data 
            wind_pdf = wind_pdf[:-1]
    
    # Make sure they all have the same dates 
    dates = wind_pdf.index
    try: 
        Q_field = Q_field.loc[dates]
        veg_hgts = veg_hgts.loc[dates]
        gap_sizes = gap_sizes.loc[dates]
        tot_folia = tot_folia.loc[dates]
    except: 
        raise ValueError("Remote sensing indicators do not have the same datetime index as the wind PDF")
    
    return Q_field, veg_hgts, gap_sizes, tot_folia, wind_pdf