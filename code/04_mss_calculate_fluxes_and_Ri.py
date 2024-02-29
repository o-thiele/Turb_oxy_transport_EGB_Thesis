import numpy as np
import pickle
import pylab as pl
import os
import matplotlib.dates as mdates
import scipy.io as ssio
from datetime import datetime
import time as timing
import pandas as pd
from matplotlib.lines import Line2D
import cmocean

import warnings
warnings.filterwarnings("ignore")
#own functions
import own_functions as own


#import metadata and make dictonary, which can be accessed via dot notation
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

md = {}
md = dotdict(md)
with open('metadata_moorings.pkl', 'rb') as f:
    meta = pickle.load(f)    
md.z17 = dotdict(meta['2017'])
md.z18 = dotdict(meta['2018'])
md.z19 = dotdict(meta['2019'])

def main(first, second, third, deep, filtered = 0):
    print('currently processing: ', first, second, third, deep, filtered)
    
    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    year, year_number = own.get_year_metadata(first, second, third)
    lon_mooring, lat_mooring = own.get_mooring_locations(first, second, third, deep)
    # operating_system = 'windoof'
    
    if first == True:
        transect_list = ['TS17.mat', 'TS18.mat', 'TS19.mat', 'TS110.mat', 'TS111.mat', 'TS112.mat', 'TS113.mat', 'TS114.mat', 'TS115.mat', 'TS116.mat']
        # transect_list =  ['TS18.mat']
        if deep == True:
            fs_up = md.z17.adcp_fs_up_deep
        else: 
            fs_up = md.z17.adcp_fs_up_shallow
        halocline_center = md.z17.halocline_center
        halocline_interval = md.z17.halocline_interval
        
    elif second == True:
        # transect_list = ['TS1_2.mat', 'TS1_3.mat', 'TS1_4.mat', 'TS1_5.mat', 'TS1_6.mat', 'TS1_7.mat', 'TS1_8.mat', 'TS1_9.mat',
                         # 'TS1_10.mat', 'TS1_11.mat', 'TS1_12.mat', 'TS1_13.mat', 'TS1_14.mat', 'TS1_15.mat', 'TS1_16.mat']
        transect_list = ['TS1_6.mat', 'TS1_7.mat', 'TS1_8.mat', 'TS1_9.mat',
                         'TS1_10.mat', 'TS1_11.mat', 'TS1_12.mat', 'TS1_13.mat', 'TS1_14.mat', 'TS1_15.mat', 'TS1_16.mat']
        fs_up = md.z18.adcp_fs_up
        fs_down = md.z18.adcp_fs_down
    
        halocline_center = md.z18.halocline_center
        halocline_interval = md.z18.halocline_interval
        # transect_list =  ['TS1_5.mat']
        
    elif third == True:
        if deep == True:
            transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8_2.mat', 'TR1-9.mat',
                         'TR1-10.mat', 'TR1-11.mat', 'TR1-12.mat', 'TR1-13.mat', 'TR1-14.mat'#, 'S106-1.mat', 'S106-2.mat',
                         #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                         ]
        else: transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8_2.mat', 'TR1-9.mat',
                                 'TR1-10.mat'#, 'TR1-11.mat', 'S106-1.mat', 'S106-2.mat',
                                 #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                                 ]
        # if plotPresentation == True:
        # transect_list = ['TR1-7.mat']
        # transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8.mat', 'TR1-9.mat', 'TR1-10.mat']
        halocline_center = md.z19.halocline_center
        halocline_interval = md.z19.halocline_interval
        fs_up = md.z19.adcp_fs_up
        fs_down = md.z19.adcp_fs_down
            
    # =============================================================================
    # for ADCP data
    # =============================================================================
    if deep == True:
        adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated.npz', allow_pickle = True)
    else:
        # adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated_down.npz', allow_pickle = True)
        adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated_down_avgd.npz', allow_pickle = True)
        
    if filtered == 1:
        if deep == True:
            adcp_data = np.load(path_ADCP_data + 'npz/adcp_low_freqs.npz', allow_pickle = True)
        else:
            # adcp_data = np.load(path_ADCP_data + 'npz/adcp_low_freqs_down.npz', allow_pickle = True)
            adcp_data = np.load(path_ADCP_data + 'npz/adcp_low_freqs_down_avgd.npz', allow_pickle = True)

    elif filtered == 2:
        if deep == True:
            adcp_data = np.load(path_ADCP_data + 'npz/adcp_high_freqs.npz', allow_pickle = True)
        else:
            # adcp_data = np.load(path_ADCP_data + 'npz/adcp_high_freqs_down.npz', allow_pickle = True)
            adcp_data = np.load(path_ADCP_data + 'npz/adcp_high_freqs_down_avgd.npz', allow_pickle = True)


            
    t_matlab_adcp = adcp_data['t_matlab']
    t_adcp = adcp_data['t']
    depth_adcp = adcp_data['depth'] 
    mdepth_adcp = adcp_data['mdepth']
    p_adcp = adcp_data['p']
    currx_adcp = adcp_data['currx']
    curry_adcp = adcp_data['curry']
    
    shear_adcp = own.calcShear(currx_adcp, curry_adcp, depth_adcp)
    
# =============================================================================
#     for indexADCP, u_adcp in enumerate(currx_adcp):
#         du_dz = -np.gradient(currx_adcp[indexADCP], depth_adcp)
#         dv_dz = -np.gradient(curry_adcp[indexADCP], depth_adcp)
#         shear_squared = du_dz[:]**2 + dv_dz[:]**2
#         # Ri, Nsquared_gsw_avgd, z_mid_gsw_avgd = calcRichardson(df_Nsquared_all_profiles['mean'], depths_Nsquared_all_profiles, shear_adcp_squared, depth)
#         # Ri_arr[indexADCP] = Ri
#         shear_adcp[indexADCP] = shear_squared
#     
#     # print(shear_adcp)
# =============================================================================
    downward_adcp = False
    try:
        adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_interpolated_down_avgd.npz', allow_pickle = True)
                
        t_matlab_adcp_down = adcp_data_down['t_matlab']
        t_adcp_down = adcp_data_down['t']
        depth_adcp_down = adcp_data_down['depth'] 
        mdepth_adcp_down = adcp_data_down['mdepth']
        p_adcp_down = adcp_data_down['p']
        currx_adcp_down = adcp_data_down['currx']
        curry_adcp_down = adcp_data_down['curry']
        shear_adcp_down = own.calcShear(currx_adcp_down, curry_adcp_down, depth_adcp_down)
# =============================================================================
#         shear_adcp_down = currx_adcp.copy()
#         
#         for indexADCP, u_adcp in enumerate(currx_adcp_down):
#             du_dz_down = -np.gradient(currx_adcp_down[indexADCP], depth_adcp_down)
#             dv_dz_down = -np.gradient(curry_adcp_down[indexADCP], depth_adcp_down)
#             shear_squared_down = du_dz_down[:]**2 + dv_dz_down[:]**2
#             # Ri, Nsquared_gsw_avgd, z_mid_gsw_avgd = calcRichardson(df_Nsquared_all_profiles['mean'], depths_Nsquared_all_profiles, shear_adcp_squared, depth)
#             # Ri_arr[indexADCP] = Ri
#             shear_adcp_down[indexADCP] = shear_squared_down
# =============================================================================
        downward_adcp = True
    except:
        print('no downward adcp data found')
        pass
    
    # =============================================================================
    #     #shear plot
    # =============================================================================
    offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh = own.getADCP_plot_definitions(first, second, third, deep)
    largeTicks = mdates.DayLocator()
    smallTicks = mdates.HourLocator(byhour=[6, 12, 18])
    dayformat = mdates.DateFormatter('%d.%b')
    
    if downward_adcp == False:
        depth_adcp = depth_adcp + offset_adcp
    else:
        depth_adcp = depth_adcp + offset_adcp_down

    
    # fig = pl.figure(1, figsize =(12,5))
    # pl.clf()
    # fig.suptitle(cruise + ' ' + mooring + ' --- Shear and other plot', fontsize=14)
    rows = 3
    cols = 1
    kw = {'height_ratios':[1,1,1]}
    fig, (ax,ax2,ax3) = pl.subplots(3,1,  gridspec_kw=kw, sharex=True, figsize = (12,10))
    ax2_twin = ax2.twinx()
    
    pl_adcp = ax.pcolormesh(adcp_data['t'], depth_adcp, shear_adcp.T,cmap=cmocean.cm.speed, shading='nearest', vmin=0,vmax=0.005)
    ax.set_ylim([ylow, yhigh])
    ax.set_ylabel('depth [m]')
    col = pl.colorbar(pl_adcp, ax = (ax,ax2,ax3), shrink = 0.3, anchor = (0,1), aspect = 12)
    col.set_label(r'$S^2 [s^{-2}]$')
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
# =============================================================================
#     #for MSS data
# =============================================================================
    # path_results_figures = 'D:/olivert/results/allCruises/mss_transects/figures/' + cruise + '/'
    path_data_npz = path_mss_interpolated + cruise + '/npz/'
    # path_results_npz = path_results_temp + 'npz/'
    # path_results_temp = path_results + 'N2_' + N_variant + 'S2_' + 'use-v-' + str(use_v) + '_var_' + str(S_variant) + '/'
    path_results_temp = path_results + 'fluxes_and_Ri/' + cruise + '_fluxes_Ri/'
    if not os.path.exists(path_results_temp) == True:
        os.makedirs(path_results_temp)
    
    path_results_fluxes = path_results + 'mss/mss_fluxes/' + cruise + '/'
    if not os.path.exists(path_results_fluxes) == True:
        os.makedirs(path_results_fluxes)
    # path_results_figures = path_results_temp + 'figures/'
    
    times_cruise = []
    fluxes_cruise = []
    fluxes_cruise_std = []
    Ri_cruise = []
    Ri_cruise_std = []
    
    for transect in transect_list:
        cutoff_low = 0.03
        cutoff_high = 0.01
        
        d_fluxes_halocline = {'filename':[], 'castnumber': [], 'halocline_index_down': [], 'halocline_index_up': [],
                              'valid': [], 'flux_O2_Shih_over_halocline': [], 'N2_halocline': []}
        df_fluxes_halocline = pd.DataFrame(data=d_fluxes_halocline)

        closest_cast = 0
        
        mss_data = np.load(path_data_npz + transect[:-4] + '_interp.npz', allow_pickle = True)
# =============================================================================
#         keys:                cast_nr, z, lon_start, lat_start, time_start, time_end,        
#                              sig0_interp, N_squared_MIX_interp, eps_MIX_interp, Reb_MIX_interp,
#                              O2_conc_gradient, flux_O2_shih, O2_conc_interp
# =============================================================================

        #get profiles closest to corresponding mooring:   
        if (np.sort(mss_data['lon_start']) == mss_data['lon_start']).all(): #driving in positive longitude direction
            for i_lon_mss, lon_mss in enumerate(mss_data['lon_start']):
                if lon_mss > lon_mooring:
                    closest_cast = i_lon_mss-1
                    # print('left to right', closest_cast)
                    break
        else:
            for i_lon_mss, lon_mss in enumerate(mss_data['lon_start']):
                if lon_mss < lon_mooring:
                    closest_cast = i_lon_mss-1
                    # print('right to left', closest_cast)
                    break
        
# =============================================================================
#         #check lon validity
# =============================================================================
        for i_lon_mss, lon_mss in enumerate(mss_data['lon_start']):
            # print(mss_data['sig0_interp'][:,i_lon_mss])
            #calculate halocline
            if (all(((x > (halocline_center - halocline_interval)) | np.isnan(x)) for x in mss_data['sig0_interp'][:,i_lon_mss])):
                # if deep == True:
                #     halocline_index_up = 0
                # else:
                halocline_index_up = -1
            else:
                halocline_index_up = (own.nearest_ind_nans(mss_data['sig0_interp'][:,i_lon_mss], halocline_center - halocline_interval))
            if (all(((x < (halocline_center + halocline_interval)) | np.isnan(x)) for x in mss_data['sig0_interp'][:,i_lon_mss])):
                # if deep == True:
                #     halocline_index_down = -1
                # else: 
                halocline_index_down = 0
                
                # print('no')
            else:
                halocline_index_down = (own.nearest_ind_nans(mss_data['sig0_interp'][:,i_lon_mss], halocline_center + halocline_interval))
            if 0 < (lon_mss - lon_mooring) < cutoff_high: 
            # if (i_lon_mss == (closest_cast + 1)):
                valid = 2
                flux_O2_over_halocline = np.nanmean(mss_data['flux_O2_shih'][halocline_index_down:halocline_index_up, i_lon_mss])
                # print('halocline indexes: ', halocline_index_up, halocline_index_down)
                try:
                    #for bad casts from meta file
                    if (i_lon_mss in (meta[year]['mss_bad_casts'][transect[:-4]])) == True:
                        valid = 0
                except:
                    pass
            elif 0 < (lon_mooring - lon_mss) <= cutoff_low:
            # elif (i_lon_mss == (closest_cast - 1)) | (i_lon_mss == closest_cast):
                valid = 1
                flux_O2_over_halocline = np.nanmean(mss_data['flux_O2_shih'][halocline_index_down:halocline_index_up, i_lon_mss])
                # print('halocline indexes: ', halocline_index_up, halocline_index_down)
                try:
                    if (i_lon_mss in (meta[year]['mss_bad_casts'][transect[:-4]])) == True:
                        valid = 0
                except:
                    pass
            else:
                valid = 0
                flux_O2_over_halocline = np.NAN
            
            if N_variant == 'mean':
                N2_halocline = np.nanmean(mss_data['N_squared_MIX_interp'][halocline_index_down:halocline_index_up, i_lon_mss])
            elif N_variant == 'sig0':
                N2_halocline = 9.81 / 1000 *(abs(np.nanmax(mss_data['sig0_interp'][halocline_index_down:halocline_index_up, i_lon_mss]) 
                                   - np.nanmin(mss_data['sig0_interp'][halocline_index_down:halocline_index_up, i_lon_mss]))
                                / abs((mss_data['z'][halocline_index_down])
                                    -(mss_data['z'][halocline_index_up])))
            
# =============================================================================
#             #other validity checks:
# =============================================================================
            if np.isnan(N2_halocline):
                valid = 0
            if np.isnan(mss_data['N_squared_MIX_interp'][halocline_index_down-1, i_lon_mss]):
                valid = 0
                            
            #save in dataframe:
            df_fluxes_halocline.loc[len(df_fluxes_halocline.index)] = [transect, i_lon_mss, halocline_index_down, halocline_index_up,
                                                                       valid, flux_O2_over_halocline, N2_halocline] 
                
        flux_O2_transect = np.nanmean(df_fluxes_halocline['flux_O2_Shih_over_halocline'])
        flux_O2_transect_std = np.nanstd(df_fluxes_halocline['flux_O2_Shih_over_halocline'])
        # flux_n = np.sum(~np.isnan((df_fluxes_halocline['flux_O2_Shih_over_halocline'])))
        # print('flux_n', flux_n)
        
        # print(flux_O2_transect)
        
        valid_true = (df_fluxes_halocline['valid'] == 1)|(df_fluxes_halocline['valid'] == 2)

        
# =============================================================================
#         #start validity plot
# =============================================================================
        # =============================================================================
# =============================================================================
#         #N2
# =============================================================================
        kw_valid = {'height_ratios':[2,2,1]}
            
        fig_valid, (ax_valid, ax_valid_adcp, ax_smart) = pl.subplots(3,1, gridspec_kw = kw_valid, sharex=True, figsize = (12,10))
        ax_valid_data = ax_valid.twinx()
        ax_valid_adcp_data = ax_valid_adcp.twinx()
        color_ax_valid_data = 'tab:green'

        # fig_valid.clf()
        # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
        # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
        Nsquared_col = ax_valid.pcolormesh(mss_data['lon_start'],  mss_data['z'] , mss_data['N_squared_MIX_interp'])
        col_valid = pl.colorbar(Nsquared_col, pad = 0.15)
        col_valid.set_label(r'$N^2 [s^{-2}]$')

        
        ax_valid_data.scatter(mss_data['lon_start'][valid_true], 
                    df_fluxes_halocline['N2_halocline'][valid_true], color = color_ax_valid_data, label = r'$N{^2}_{{halo}}$ valid')
        
        ax_valid_data.scatter(mss_data['lon_start'][df_fluxes_halocline['valid']==0], 
                    df_fluxes_halocline['N2_halocline'][df_fluxes_halocline['valid']==0], color = 'tab:red', label = r'$N{^2}_{{halo}}$ not valid')
        
        # ax_valid.scatter(mss_data['lon_start'][df_fluxes_halocline['valid']==2], 
        #             np.full(len(mss_data['lon_start'][df_fluxes_halocline['valid']==2]), fill_value = -20), color = 'tab:green')

        #halocline limits
        ax_valid.step(mss_data['lon_start'], mss_data['z'][df_fluxes_halocline['halocline_index_down']], color = 'tab:red', where = 'mid', label = 'halocline')
        ax_valid.step(mss_data['lon_start'], mss_data['z'][df_fluxes_halocline['halocline_index_up']], color = 'tab:red', where = 'mid')
        #N2
        # ax_valid.scatter
        
        # ax_valid.set_xlabel('longitude at cast start')
        ax_valid.set_ylabel('depth [m]')
        ax_valid.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
        ax_valid.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
        # pl.legend(loc = 'upper left')
        # pl.tight_layout()
        
        ax_valid_data.set_ylabel(r'$N{^2}_{{halo}} [s^{-2}]$')
        ax_valid_data.set_ylim(bottom=0)
        ax_valid_data.tick_params(axis='y')#, labelcolor=color_ax_valid_data)   
        ax_valid_data.legend(loc = 'upper left')
        
# =============================================================================
#         #adcp, S2:
# =============================================================================
        # print(mss_data['time_start'])
        shear_used = np.zeros((len(mss_data['time_start']), len(adcp_data['depth'])))
        #to calculate shear from velocities over and below the halocline and use delta z from the mss
        currx_used = shear_used.copy()
        curry_used = currx_used.copy()
        
        shear_halocline = np.zeros(len(mss_data['time_start']))
        adcp_depths_down = np.zeros(len(mss_data['time_start']))
        adcp_depths_up = np.zeros(len(mss_data['time_start']))
        
        if S_variant == 3:
            #variant 3: use min and max velocities in largest valid adcp boundaries 

            ind_adcp_depth_down_largest = 0
            ind_adcp_depth_up_largest = 0
            for count, mss_time_start in enumerate(mss_data['time_start']):
                ind_adcp_depth_down_tmp = own.nearest_ind_nans(depth_adcp, mss_data['z'][df_fluxes_halocline['halocline_index_down']][count])
                ind_adcp_depth_up_tmp = own.nearest_ind_nans(depth_adcp, mss_data['z'][df_fluxes_halocline['halocline_index_up']][count])
                if (abs(depth_adcp[ind_adcp_depth_down_tmp] - depth_adcp[ind_adcp_depth_up_tmp]) > abs(depth_adcp[ind_adcp_depth_down_largest] - depth_adcp[ind_adcp_depth_up_largest])) & (df_fluxes_halocline['valid'][count] != 0):
                    ind_adcp_depth_down_largest = ind_adcp_depth_down_tmp
                    ind_adcp_depth_up_largest = ind_adcp_depth_up_tmp
            # print(ind_adcp_depth_down_largest, ind_adcp_depth_up_largest)
        
        for count, mss_time_start in enumerate(mss_data['time_start']):
            # timestamp_mss = pd.to_datetime(mss_time_start)
            ts = (mss_time_start - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            timestamp_mss = datetime.utcfromtimestamp(ts)
            # print(timestamp_mss)
            # print(t_adcp, timestamp_mss)
            indexADCP = own.nearest_ind(t_adcp, timestamp_mss)
            
            if S_variant == 1:            
                #variant 1: use shear and closest cast halocline adcp boundaries 
                shear_used[count,:] = shear_adcp[indexADCP,:]
                # get z depths and indexes from halocline for adcp:
                ind_adcp_depth_down = own.nearest_ind_nans(depth_adcp, mss_data['z'][df_fluxes_halocline['halocline_index_down']][closest_cast])
                ind_adcp_depth_up = own.nearest_ind_nans(depth_adcp, mss_data['z'][df_fluxes_halocline['halocline_index_up']][closest_cast])
                
                if ind_adcp_depth_up < ind_adcp_depth_down :
                    # print('changing index')
                    ind_temp = ind_adcp_depth_down
                    ind_adcp_depth_down = ind_adcp_depth_up
                    ind_adcp_depth_up = ind_temp
                
                adcp_depths_down[count] = depth_adcp[ind_adcp_depth_down]
                adcp_depths_up[count] = depth_adcp[ind_adcp_depth_up]

                shear_halocline[count] = np.nanmean(shear_used[count,ind_adcp_depth_down:ind_adcp_depth_up])
            
            elif S_variant == 2:
            #variant 2: use velocities for shear calculation and flexible adcp boundaries from mss halocline:
                if use_v == True:
                    currx_used[count,:] = currx_adcp[indexADCP,:]
                    # print(currx_used[count,:])
                    curry_used[count,:] = curry_adcp[indexADCP,:]
                    ind_adcp_depth_down = own.nearest_ind_nans(depth_adcp, mss_data['z'][df_fluxes_halocline['halocline_index_down']][count])
                    ind_adcp_depth_up = own.nearest_ind_nans(depth_adcp, mss_data['z'][df_fluxes_halocline['halocline_index_up']][count])
                    
                    if ind_adcp_depth_up < ind_adcp_depth_down :
                        # print('changing index')
                        ind_temp = ind_adcp_depth_down
                        ind_adcp_depth_down = ind_adcp_depth_up
                        ind_adcp_depth_up = ind_temp
                    
                    if ind_adcp_depth_up != (len(currx_used[count,:])-1):
                        # print(ind_adcp_depth_up, len(currx_used[count,:]))
                        ind_adcp_depth_up = ind_adcp_depth_up + 1
                    
                    # print(adcp_depths_up, adcp_depths_down)
    
                    adcp_depths_down[count] = depth_adcp[ind_adcp_depth_down]
    
                    adcp_depths_up[count] = depth_adcp[ind_adcp_depth_up]
    
                        
                    # print(adcp_depths_up, adcp_depths_down)
    
                    
                    if ((((mss_data['z'][df_fluxes_halocline['halocline_index_down']][count])
                       -(mss_data['z'][df_fluxes_halocline['halocline_index_up']][count])) != 0) and 
                        (((abs(currx_used[count, ind_adcp_depth_up] - currx_used[count, ind_adcp_depth_down]) 
                                             + abs(curry_used[count, ind_adcp_depth_up] - curry_used[count, ind_adcp_depth_down]))) != 0)):
                        # print('jo')
                        
                        #TODO: check shear calculation!!
    # =============================================================================
    #   # velocity in upper bin and lower bin:
    #                     shear_halocline[count] = ((abs(currx_used[count, ind_adcp_depth_up] - currx_used[count, ind_adcp_depth_down]) 
    #                                           + abs(curry_used[count, ind_adcp_depth_up] - curry_used[count, ind_adcp_depth_down]))
    #                                           / ((mss_data['z'][df_fluxes_halocline['halocline_index_down']][count])
    #                                              -(mss_data['z'][df_fluxes_halocline['halocline_index_up']][count])))**2
    # =============================================================================
    #   max and min velocites between upper and lower bins:
                        shear_halocline[count] = ((abs(np.nanmax(currx_used[count, ind_adcp_depth_down:ind_adcp_depth_up]) - np.nanmin(currx_used[count, ind_adcp_depth_down:ind_adcp_depth_up])) 
                                              + abs(np.nanmax(curry_used[count, ind_adcp_depth_down:ind_adcp_depth_up]) - np.nanmin(curry_used[count, ind_adcp_depth_down:ind_adcp_depth_up])))
                                              / ((mss_data['z'][df_fluxes_halocline['halocline_index_down']][count])
                                                 -(mss_data['z'][df_fluxes_halocline['halocline_index_up']][count])))**2
                        # print(shear_halocline[count])
                    #new variant 2
                    else: 
                        shear_halocline[count] = np.NaN
                    if shear_halocline[count] == 0:
                        shear_halocline[count] = np.NaN
                    #end new variant 2

            elif S_variant == 3:
            #variant 3: use min and max velocities in largest valid adcp boundaries 
                if use_v == True:
                    currx_used[count,:] = currx_adcp[indexADCP,:]
                    # print(currx_used[count,:])
                    curry_used[count,:] = curry_adcp[indexADCP,:]
                    ind_adcp_depth_down = ind_adcp_depth_down_largest
                    ind_adcp_depth_up = ind_adcp_depth_up_largest
                    
                    if ind_adcp_depth_up < ind_adcp_depth_down :
                        # print('changing index')
                        ind_temp = ind_adcp_depth_down
                        ind_adcp_depth_down = ind_adcp_depth_up
                        ind_adcp_depth_up = ind_temp
                    
                    if ind_adcp_depth_up != (len(currx_used[count,:])-1):
                        # print(ind_adcp_depth_up, len(currx_used[count,:]))
                        ind_adcp_depth_up = ind_adcp_depth_up + 1
                    
                    # print(adcp_depths_up, adcp_depths_down)
    
                    adcp_depths_down[count] = depth_adcp[ind_adcp_depth_down]
    
                    adcp_depths_up[count] = depth_adcp[ind_adcp_depth_up]
    
                        
                    # print(adcp_depths_up, adcp_depths_down)
    
                    
                    if ((((mss_data['z'][df_fluxes_halocline['halocline_index_down']][count])
                       -(mss_data['z'][df_fluxes_halocline['halocline_index_up']][count])) != 0) and 
                        (((abs(currx_used[count, ind_adcp_depth_up] - currx_used[count, ind_adcp_depth_down]) 
                                             + abs(curry_used[count, ind_adcp_depth_up] - curry_used[count, ind_adcp_depth_down]))) != 0)):
    
    #   max and min velocites between upper and lower bins:
                        shear_halocline[count] = ((abs(np.nanmax(currx_used[count, ind_adcp_depth_down:ind_adcp_depth_up]) - np.nanmin(currx_used[count, ind_adcp_depth_down:ind_adcp_depth_up])) 
                                              + abs(np.nanmax(curry_used[count, ind_adcp_depth_down:ind_adcp_depth_up]) - np.nanmin(curry_used[count, ind_adcp_depth_down:ind_adcp_depth_up])))
                                              / ((mss_data['z'][df_fluxes_halocline['halocline_index_down']][count])
                                                 -(mss_data['z'][df_fluxes_halocline['halocline_index_up']][count])))**2

                    else: 
                        shear_halocline[count] = np.NaN
                    if shear_halocline[count] == 0:
                        shear_halocline[count] = np.NaN
        
                # print(shear_halocline[count])
        # ax_valid_adcp.pcolormesh(adcp_data['t'], depth_adcp, shear_adcp.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=0,vmax=0.005)
        if use_v == False:
            Ssquared_col = ax_valid_adcp.pcolormesh(mss_data['lon_start'], depth_adcp, shear_used.T, shading='nearest', vmin=0,vmax=0.005)
            col_valid_s2 = pl.colorbar(Ssquared_col, pad = 0.15,  ax = (ax_valid_adcp, ax_smart), shrink = 0.60, anchor = (0,1), aspect = 15)
            col_valid_s2.set_label(r'$S^2 [s^{-2}]$')

        else:
            Ssquared_col = ax_valid_adcp.pcolormesh(mss_data['lon_start'], depth_adcp, curry_used.T, shading='nearest', cmap=cmocean.cm.balance , vmin=-0.25,vmax=0.25)
            col_valid_s2 = pl.colorbar(Ssquared_col, pad = 0.15,  ax = (ax_valid_adcp, ax_smart), shrink = 0.60, anchor = (0,1), aspect = 15)
            col_valid_s2.set_label(r'$v [ms^{-1}]$')
        # ax_valid_adcp.scatter(mss_data['lon_start'][closest_cast], z_zu_idx, color = colors[1])
        ax_valid_adcp.step(mss_data['lon_start'], mss_data['z'][df_fluxes_halocline['halocline_index_down']], color = 'tab:red', where = 'mid')
        ax_valid_adcp.step(mss_data['lon_start'], mss_data['z'][df_fluxes_halocline['halocline_index_up']], color = 'tab:red', where = 'mid')
        ax_valid_adcp.step(mss_data['lon_start'], adcp_depths_down, color = 'tab:green', where = 'mid')
        ax_valid_adcp.step(mss_data['lon_start'], adcp_depths_up, color = 'tab:green', where = 'mid')
        ax_valid_adcp.set_ylabel('depth [m]')



        ax_valid_adcp_data.scatter(mss_data['lon_start'][valid_true], shear_halocline[valid_true], 
                                   color = color_ax_valid_data,  label = r'$S{^2}_{{halo}}$ valid')
        # ax_valid_adcp_data.scatter(mss_data['lon_start'][df_fluxes_halocline['valid'] == 2], shear_halocline[df_fluxes_halocline['valid'] == 2], color = color_ax_valid_data)

        ax_valid_adcp_data.scatter(mss_data['lon_start'][df_fluxes_halocline['valid'] == 0], 
                                   shear_halocline[df_fluxes_halocline['valid'] == 0], 
                                   color = 'tab:red',  label = r'$S{^2}_{{halo}}$ not valid')
        
        ax_valid_adcp_data.legend(loc = 'upper left')

        ax_valid_adcp_data.set_ylabel(r'$S{^2}_{halo} [s^{-2}]$')
        ax_valid_adcp_data.set_ylim(bottom=0)
        ax_valid_adcp_data.tick_params(axis='y')#, labelcolor=color_ax_valid_data)   
        # ax_valid_adcp.set_xlim(mss_data['time_start'][0], mss_data['time_end'][-1])
        
        ax_smart.scatter(mss_data['lon_start'], shear_halocline, color = 'tab:orange', label = r'$S{^2}_{halo}$')
        ax_smart.scatter(mss_data['lon_start'][df_fluxes_halocline['valid']==1], 
                    df_fluxes_halocline['N2_halocline'][df_fluxes_halocline['valid']==1], color = 'tab:blue', label = r'$N{^2}_{halo}$')
        
        ax_smart.scatter(mss_data['lon_start'][df_fluxes_halocline['valid']==2], 
                    df_fluxes_halocline['N2_halocline'][df_fluxes_halocline['valid']==2], color = 'tab:blue')
        ax_smart.grid()
        ax_smart.legend(loc = 'upper left')
        ax_smart.set_ylim(bottom=0)
        ax_smart.set_xlabel('longitude at cast start')
        # col_valid_smart = pl.colorbar(Ssquared_col, pad = 0.1)
    
        # fig_valid.savefig(path_results_temp + cruise + '_' + '_' + mooring + transect[:-4] + '_validity_filtered{}.png'.format(filtered), dpi = 300)

        
        fig_valid.clf()
        
# =============================================================================
#         #end validity plots
# =============================================================================

# =============================================================================
# #start oxygen plot
# =============================================================================
        kw_oxy = {'height_ratios':[1]}
            
        fig_oxy, ax_oxy = pl.subplots(1,1, gridspec_kw = kw_oxy, figsize = (12,6))

        oxy_col = ax_oxy.pcolormesh(mss_data['lon_start'],  mss_data['z'] , mss_data['flux_O2_shih'], cmap = cmocean.cm.balance, vmin = -10, vmax = 10)
        col_valid = pl.colorbar(oxy_col)
        col_valid.set_label(r'$ F_{O2} [mmol \ m^{-2}d^{-1}]$')

        ax_oxy.scatter(mss_data['lon_start'], 
                    -df_fluxes_halocline['N2_halocline']/df_fluxes_halocline['N2_halocline'], color = 'tab:red', label = 'cast not valid')
        
        ax_oxy.scatter(mss_data['lon_start'][valid_true], 
                    -df_fluxes_halocline['N2_halocline'][valid_true]/df_fluxes_halocline['N2_halocline'][valid_true], color = 'tab:green', label = 'cast valid')
        
        CS = pl.contour(mss_data['lon_start'], mss_data['z'], mss_data['sig0_interp'], 12, colors = 'k', alpha = 0.5)
        pl.clabel(CS, inline = 1)

        #halocline limits
        ax_oxy.step(mss_data['lon_start'], mss_data['z'][df_fluxes_halocline['halocline_index_down']], color = 'goldenrod', where = 'mid', label = 'halocline')
        ax_oxy.step(mss_data['lon_start'], mss_data['z'][df_fluxes_halocline['halocline_index_up']], color = 'goldenrod', where = 'mid')

        ax_oxy.set_xlabel('lon at cast start')
        ax_oxy.set_ylabel('depth [m]')
        ax_oxy.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'deep mooring')
        ax_oxy.scatter(meta[year]['lon_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange')

        # ax_oxy.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')

        ax_oxy.legend()
        if filtered == 0:
            # fig_oxy.savefig(path_results_fluxes + cruise + '_' + mooring + '_' + transect[:-4] + '_oxy.png', dpi = 300, bbox_inches = 'tight')
            pass
        
        fig_oxy.clf()

        
        # print(df_fluxes_halocline)
        # pl.subplot(rows,cols,2, adjustable='box', sharex = ax)
        
        colors_ax2 = ['gold' , 'tab:orange', 'tab:red']
        
        ax2.scatter(mss_data['time_start'][df_fluxes_halocline['valid']==1], 
                    df_fluxes_halocline['flux_O2_Shih_over_halocline'][df_fluxes_halocline['valid']==1], color = colors_ax2[0])
        ax2.scatter(mss_data['time_start'][df_fluxes_halocline['valid']==2], 
                    df_fluxes_halocline['flux_O2_Shih_over_halocline'][df_fluxes_halocline['valid']==2], color = colors_ax2[1])
        ax2.scatter(mss_data['time_start'][closest_cast], df_fluxes_halocline['flux_O2_Shih_over_halocline'][closest_cast], color = colors_ax2[2])

        #mean fluxes
        color_twin = 'teal'
        ax2_twin.scatter(mss_data['time_start'][closest_cast], 
                    flux_O2_transect, color = color_twin, marker = 'X')
                
        #N2 and S2 and Ri:
        ax3.scatter(mss_data['time_start'][valid_true],
                    (df_fluxes_halocline['N2_halocline'][valid_true])/shear_halocline[valid_true], 
                    color = colors_ax2[0])

        #mean value
        Ri_mean = np.nanmean((df_fluxes_halocline['N2_halocline'][valid_true])/shear_halocline[valid_true])
        Ri_std = np.nanstd((df_fluxes_halocline['N2_halocline'][valid_true])/shear_halocline[valid_true])
        Ri_n = np.sum(~np.isnan((df_fluxes_halocline['N2_halocline'][valid_true])/shear_halocline[valid_true]))
        # print(Ri_n)
        ax3.scatter(mss_data['time_start'][closest_cast], 
                    Ri_mean, 
                    color = color_twin)
        
        times_cruise.append(mss_data['time_start'][closest_cast])
        
        fluxes_cruise.append(flux_O2_transect)
        fluxes_cruise_std.append(flux_O2_transect_std/np.sqrt(Ri_n))
        Ri_cruise.append(Ri_mean)
        Ri_cruise_std.append(Ri_std/np.sqrt(Ri_n))  #not std, but standard deviation of the mean
    
    np.savez(path_results_temp + cruise + mooring + 'fluxes_Ri_cruise_filtered_{}'.format(filtered), 
             fluxes_cruise = fluxes_cruise, fluxes_cruise_std = fluxes_cruise_std, Ri_cruise = Ri_cruise, Ri_cruise_std = Ri_cruise_std, time_start_closest = times_cruise)

    ax2.grid(visible = True)
    legend_elements_ax2 = [Line2D([0], [0], marker='o', color = colors_ax2[0], label = 'lon < lon_mooring'), 
                    Line2D([0], [0], marker='o', color = colors_ax2[1], label = 'lon > lon_mooring'),
                    Line2D([0], [0], marker='o', color = colors_ax2[2], label = 'closest cast')]
    ax2.legend(handles=legend_elements_ax2, loc='lower left')    
    ax2.set_ylabel(r'$F_{O2} [mmol \ m^{-2}d^{-1}]$')
    ax2_twin.set_ylabel(r'$\langle F_{O2} \rangle [mmol \ m^{-2}d^{-1}]$', color = color_twin)
    ax2_twin.tick_params(axis='y', labelcolor=color_twin)   
    
    
    ax3.set_ylabel(r'$Ri[]$')
    ax3.grid(visible = True)
    legend_elements_ax3 = [Line2D([0], [0], marker='o', color = colors_ax2[0], label = r'$Ri$'), 
                    Line2D([0], [0], marker='o', color = color_twin, label = r'$\langle Ri \rangle$'),
                    Line2D([0], [0], ls = '--', color = color_twin, label = '0.25')]
    ax3.legend(handles=legend_elements_ax3, loc='upper left')
    xlims = ax3.get_xlim()
    ax3.hlines(0.25, xmin = xlims[0], xmax = xlims[1], ls = '--', color = color_twin)
    ax3.set_yscale('log')

        
    # pl.tight_layout()
    # fig.savefig(path_results_temp + cruise + mooring + 'flux_Ri_filtered{}.png'.format(filtered),dpi=300)
    # pl.show()       #slow for large pcolormeshs
# =============================================================================
#     fig.clf()
#     ax2.cla()
#     ax2_twin.cla()
# =============================================================================
    
starttime = timing.time()

S_variant_list = [1,2,3]
N_variant_list = ['mean', 'sig0']

S_variant_list = [2]
N_variant_list = ['sig0']

filtered_list = [0, 1, 2]
    #0: unfiltered
    #1: low-pass
    #2: high-pass
    
# cruise_years = ['2019']
cruise_years = ['2017', '2018', '2019']

for N_variant in N_variant_list:
    for S_variant in S_variant_list:
            if (S_variant == 1):
                use_v = False
            else:
                use_v = True
                for cruise_year in cruise_years:
                    if cruise_year == '2017':
                        for filtered in filtered_list:
                            main(True, False, False, True, filtered)
                    elif cruise_year == '2018':
                        for filtered in filtered_list:
                            main(False, True, False, True, filtered)
                    elif cruise_year == '2019':
                        for filtered in filtered_list:
                            main(False, False, True, True, filtered)
                            main(False, False, True, False, filtered)

# =============================================================================
# for N_variant in N_variant_list:
#     for S_variant in S_variant_list:
#             if (S_variant == 1):
#                 use_v = False
#             else:
#                 use_v = True
# 
#             for cruise_year in cruise_years:
#                 if cruise_year == '2017':
#                     main(True, False, False, True)
#                     main(True, False, False, True, 1)    #for filtered data
#                     main(True, False, False, True, 2)    #for filtered data
#                     # main(True, False, False, False)
#                 elif cruise_year == '2018':
#                     main(False, True, False, True)
#                     main(False, True, False, True, 1)    #for filtered data
#                     main(False, True, False, True, 2)    #for filtered data
#                     # main(False, True, False, False)
#                 elif cruise_year == '2019':
#                     main(False, False, True, True)
#                     main(False, False, True, True, 1)    #for filtered data
#                     main(False, False, True, True, 2)    #for filtered data
#                     # main(False, False, True, False)
#                     # main(False, False, True, False, 1)    #for filtered data
#                     # main(False, False, True, False, 2)    #for filtered data
# =============================================================================

endtime = timing.time()
print(endtime - starttime)
