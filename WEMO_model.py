"""WEMO_model.py 

    Python module containing WEMO model function that computes a predicted horizontal flux (Q pred) from various input data and parameter values. Allows for computation of best-fit parameter values. 
    
    Author: Nicole Keeney 
    Date Created: 06-17-2021 
    Modification History: 
        - Changed to allow input of total foliage by date, instead of one static total foliage (Fg) value (06-29-2021)
        - Input different equations for Q(t,pred) (07-20-2021)
        - Updated wind/ustar pdf workflow (02-22-2022)
"""

import pandas as pd 
import numpy as np


def WEMO_model(scl_hgt, Pd_xh, ustar_pdf, q_pred_eq="mod_shao", Fg=0.3, A=10**(-4.246621), C=5.155137, U=0.048653, z0=0.079, u_threshold=0.3): 

    """ Run WEMO model for a set of parameters and meteorological/ecological inputs
    You can set the argument q_pred_eq to indicate which equation to use to compute Q(t,pred). See Li et al (2013) table 4 and the argument description for more information. 
    
    Args: 
        scl_hgt (pd.DataFrame): scale height by gap size (columns) for each date (index)
        Pd_xh (pd.DataFrame): gap size probability density function with gap sizes (columns) for each date (index) 
        ustar_pdf (pd.DataFrame): wind probability density function (columns) by date (index)
        q_pred_eq (select one of "owen","mod_shao", or "sorenson"; optional): 
            - "mod_shao": Use the modified Shao et al (1993) equation to compute Q(t,pred)? (default to "mod_shao")
            - "owen": Use the Owen (1964), Shao et al (1993), Gillette & Chen (2001) equation to compute Q(t,pred)? 
            - "sorenson": Use the Sorenson 1991 equation to compute Q(t,pred)? (default to False. 
        A (float, optional): constant with variable units 
        C (float, optional): e-folding distance for recovery of the shear stress in the lee of plants, dimensionless
        U (float, optional): shear velocity ratio in the immediate lee of a plant, dimensionless
        z0 (float, optional): roughness length (m)
        Fg (pd.DataFrame or float, optional): ground fraction that is covered by vegetation (allows for Fg by date, or one static Fg value) (default to 0.3)
        u_threshold (float, optional): threshold wind velocity (m/s) (default to 0.3 m/s)
   
    Returns: 
        Q_pred_date_df (pd.DataFrame): predicted horizontal flux per date in scl_hgt pd.DataFrame
    
    """
    # Constants 
    rho = 1225   # Density of dry air, g/m3
    g = 9.81  # Acceleration due to gravity, m/s2
    
    # Get each ustar bin value from column names of PDF 
    u_star = ustar_pdf.columns.values

    # Loop through dates
    wemo_dates = ustar_pdf.index
    Q_pred_date = []
    for date in wemo_dates: 
        # Compute u*s/u*
        x_h = scl_hgt.loc[date].T.to_numpy() # x_sc_gap_array in Abi's code. This gives you the scale height at each gap size for a given date (a list wuth 4 values)
        shear_velocity_ratio = (U + ((1 - U)*(1 - np.exp(-C/x_h)))) # shear velocity ratio is u*s/u*
        Pd_xh_np = Pd_xh.loc[date].to_numpy() # Probability that any point in the landscape is a certain distance from the nearest upwind plant expressed in units of height of that plant
        
        # Deal with foliage input if it's a dataframe. Some checks here to make sure user inputted a valid value 
        if type(Fg) == pd.DataFrame: 
            Fg_date = Fg.loc[date][0]
        elif type(Fg) != float: 
            raise ValueError("Fg input needs to be a pandas DataFrame with a datetime index the same as the other input datasets, or a float (for a static fraction)")
        if (Fg_date > 1) and (Fg_date <= 100): 
            Fg_date = Fg_date/100 # Make it a fraction 0-1
        if (Fg_date < 0) or (Fg_date >100): 
            raise ValueError("The ground cover fraction needs to be a float between 0-1 or 0-100") 

        q_xh_list, Q_t_ustar_list, Q_pred_list = [],[],[]
        for u_star_i in u_star: # Loop through each shear velocity bin 

            # Compute u*s, the shear velocity downwind of a plant at each gap size  
            u_star_s = u_star_i * shear_velocity_ratio

            # Compute horizontal flux at the date, for each gap size
            if q_pred_eq.lower()=="owen": # Owen 1964 / Shao 1993 / Gillette & Chen 2001
                q_xh = A*(rho/g)*(u_star_s**3)*(1-((u_threshold**2)/(u_star_s**2))) 
            elif q_pred_eq.lower()=="sorenson": # Sorenson 1991
                q_xh = A*(rho/g)*(u_star_s**3)*(1-(u_threshold/u_star_s))*(1+17.75*(u_threshold/u_star_s))
            elif q_pred_eq.lower()=="mod_shao": # Modified Shao 1993 
                q_xh = A*(rho/g)*(u_star_s**2)*(1-((u_threshold**2)/(u_star_s**2)))
            else: # Raise error if bad input
                raise ValueError("You must input one value of owen, mod_shao, or sorenson to indicate the equation for Q(t,pred) to use")

            # Replace any negative values with 0 
            q_xh[q_xh < 0] = 0

            # Compute horizontal aeolian flux Qtu for a specific wind shear velocity u*
            Q_t_ustar = (1-Fg_date)*np.nansum(np.dot(q_xh, Pd_xh_np)) # Eq 3, Li et al (2013). Horizontal flux for wind shear velocity u_star_i
            Q_t_ustar_list.append(Q_t_ustar) 

        # Dot with the wind shear distribution over all dates 
        # This gives you Pu*, the probability distribution of wind shear velocity over the entire study period
        P_u_star = ustar_pdf.loc[date]
        Q_t_pred = np.nansum(Q_t_ustar_list*P_u_star.values) # Eq 4, Li et al (2013)
        Q_pred_date.append(Q_t_pred*86400) # g/m/day
        
    # Save results as dataframe
    Q_pred_date_df = pd.DataFrame(np.array(Q_pred_date), index=wemo_dates.values, columns=["Q_pred"])
    Q_pred_date_df.index.name = 'Date'
    return(Q_pred_date_df)