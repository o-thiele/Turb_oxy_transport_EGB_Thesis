import numpy as np
import pylab as pl
import os
import matplotlib.dates as mdates
import time as timing
import scipy.io as ssio
import pandas as pd
from scipy.interpolate import interp1d
import pickle
import cmocean

import own_functions as own

# =============================================================================
MINI_SIZE = 14
SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 16
pl.rc('font', size=SMALL_SIZE)          # controls default text sizes
pl.rc('axes', titlesize=SMALL_SIZE, titleweight = "bold")     # fontsize of the axes title
pl.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
pl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pl.rc('legend', fontsize=MINI_SIZE)    # legend fontsize
pl.rc('figure', titlesize=MEDIUM_SIZE, titleweight = "bold")  # fontsize of the figure title

starttime = timing.time()

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

# =============================================================================
# first = True
# first = False
# second = True
# second = False
# third = True
# # third = False
# 
# deep = True
# =============================================================================

def main(first, second, third, deep):

    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    # operating_system = 'windoof'
    year, year_number = own.get_year_metadata(first, second, third)
    lon_mooring, lat_mooring = own.get_mooring_locations(first, second, third, deep)
    operating_system = 'linux'
    
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
        transect_list = ['TS1_2.mat', 'TS1_3.mat', 'TS1_4.mat', 'TS1_5.mat', 'TS1_6.mat', 'TS1_7.mat', 'TS1_8.mat', 'TS1_9.mat',
                         'TS1_10.mat', 'TS1_11.mat', 'TS1_12.mat', 'TS1_13.mat', 'TS1_14.mat', 'TS1_15.mat', 'TS1_16.mat']
        fs_up = md.z18.adcp_fs_up
        fs_down = md.z18.adcp_fs_down
    
        halocline_center = md.z18.halocline_center
        halocline_interval = md.z18.halocline_interval
        # transect_list =  ['TS1_10.mat']
        
    elif third == True:
        # if plotPresentation == True:
        #     transect_list = ['TR1-2.mat', 'TR1-9.mat']
        transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8.mat', 'TR1-9.mat',
                     'TR1-10.mat', 'TR1-11.mat', 'TR1-12.mat', 'TR1-13.mat', 'TR1-14.mat'#, 'S106-1.mat', 'S106-2.mat',
                     #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                     ]
        # transect_list = ['TR1-8.mat']
        halocline_center = md.z19.halocline_center
        halocline_interval = md.z19.halocline_interval
        fs_up = md.z19.adcp_fs_up
        fs_down = md.z19.adcp_fs_down
    
    # path_results_figures = 'D:/olivert/results/allCruises/mss_transects/figures/' + cruise + '/'
    path_results_temp = path_mss_interpolated + cruise + '/'
    path_results_npz = path_results_temp + 'npz/'
    path_results_figures = path_results_temp + 'figures/'
    
    if not os.path.exists(path_results_npz) == True:
        os.makedirs(path_results_npz)
    if not os.path.exists(path_results_figures) == True:
        os.makedirs(path_results_figures)
        
    # =============================================================================
    # MSS Data
    # =============================================================================
    #loop through all files in corresponding cruise
    
    zmin = -125
    zmax = 0
    dz = 0.1    #old 0.25
    z = np.arange(zmin,zmax,dz)     #interpolation z-levels
    
    flux_O2_shih_all = []
    N2_halocline = []
    z_N2_up = []
    z_N2_down = []
    
    for filename in transect_list:
        #initialize arrays to save data:
        d_mss = {'filename':[], 'castnumber': [], 'lat_start': [], 'lon_start': [], 'time_start': [], 'time_end': []}
        df_mss = pd.DataFrame(data=d_mss)
        #load matlab data:
        matlab = ssio.loadmat(path_to_mss_files + filename)
        DATA_sub = matlab['DATA']
        CTD_sub = matlab['CTD']
        MIX_sub = matlab['MIX']
        
        halocline_index_up = []
        halocline_index_down = []
        flux_O2_over_halocline = []
        flux_O2_for_cruise = 0
        
    # =============================================================================
    #     #EMB217 - topographic waves
    # =============================================================================
        
        if third == True:
              # O2_CTD = CTD_sub['O2'][0][count]
              O2raw_interp = own.interp_MSS_var(DATA_sub, z, 'rawO2')   #TODO: which unit?
              O2_sat_interp = own.interp_MSS_var(CTD_sub, z, 'O2')  
              P_CTD = own.interp_MSS_var(CTD_sub, z, 'P')  
              SA_CTD = own.interp_MSS_var(CTD_sub, z, 'SA')
              CT_CTD = own.interp_MSS_var(CTD_sub, z, 'CT')  
              lat = md.z19.lat_deep
              lon = md.z19.lon_deep
              O2_conc_interp = own.calc_O2conc(O2_sat_interp, P_CTD, SA_CTD, CT_CTD, lon, lat)
              sig0_interp = own.interp_MSS_var(CTD_sub, z, 'SIGTH')  
    
              NProf = len(CTD_sub['P'][0])
              
        # =============================================================================
        #         #EMB 169 and EMB 177
        # =============================================================================
        elif first == True or second == True:      
            matlab_oxyshift = ssio.loadmat(path_to_mss_files + 'oxy_shift/' + filename[:-4] + '_TODL_merged_shift_oxy.mat')
            TODL_sub = matlab_oxyshift['TODL_MSS_oxy']
    
            #loop over all casts and append them to matrix    
            
            NProf = len(TODL_sub['P'][0])
            # number of profiles
        
            # Initialize the sigma and O2 arrays to store interpolated density and oxygen values
            O2_interp = np.zeros((len(z), NProf)) # z is the depth, NProf is the number of profiles
            O2_conc_interp = O2_interp.copy()
    
            sig0_interp = O2_interp.copy()
        
        # =============================================================================
        #     #for all cruises
        # =============================================================================
        N_squared_MIX_interp = own.interp_MSS_var(MIX_sub, z, 'N2')
        eps_MIX_interp = own.interp_MSS_var(MIX_sub, z, 'eps')
        Reb_MIX_interp = own.interp_MSS_var(MIX_sub, z, 'Re')
    
        
        turbulent_diffusivity_Shih_interp = own.get_turbulent_diffusivity_Shih(Reb_MIX_interp, eps_MIX_interp, N_squared_MIX_interp)
        turbulent_diffusivity_Shih_interp_matlab = own.interp_MSS_var(MIX_sub, z, 'KrhoShih')
        flux_O2_shih = turbulent_diffusivity_Shih_interp.copy()
        O2_conc_gradient_whole = turbulent_diffusivity_Shih_interp.copy()
        
        for i in range(NProf): # loop over profiles
            # depth
            lat_mss = matlab['STA'][0, i][3]
            lon_mss = matlab['STA'][0, i][4]
            # start, end = calcStartAndEndtime(groundpath_mss + finepath_mss_matlab + filename, i)  #very time consuming!!!
            start, end = own.calcStartAndEndtime_singleCast(matlab, i, operating_system)
            df_mss.loc[len(df_mss.index)] = [filename, i, lat_mss[0][0], lon_mss[0][0], start, end] 
            
            if first or second:
                O2_TODL, z_TODL, sig0_TODL, O2_mumol = own.load_O2_shifted(path_to_mss_files, filename, i)
                # ZTmp = -TODL_sub['P'][0][i] 
                ZTmp = z_TODL
                Ind = ~np.isnan(ZTmp) # find the index of non-nan values
                # print(Ind)
                ZTmp = ZTmp[Ind] # remove nan values
                # Var = TODL_sub['oxy'][0][i][:,4][Ind[0]] #varinterp is the variable to be interpolated, Ind is the index of non-nan values
                O2Tmp = O2_TODL[Ind] 
                sig0Tmp = sig0_TODL[Ind]          
                O2_conc_Tmp = O2_mumol[Ind[0]] 
        
                # Perform the interpolation
                interp_func = interp1d(ZTmp, O2Tmp, bounds_error=False, fill_value=np.nan)
                O2_interp[:, i] = interp_func(z) 
                
                interp_func = interp1d(ZTmp, O2_conc_Tmp, bounds_error=False, fill_value=np.nan)
                O2_conc_interp[:, i] = interp_func(z) 
                
                interp_func = interp1d(ZTmp, sig0Tmp, bounds_error=False, fill_value=np.nan)
                sig0_interp[:, i] = interp_func(z) 
            
            #calculate halocline
            halocline_index_up.append(own.nearest_ind_nans(sig0_interp[:,i], halocline_center - halocline_interval))
            halocline_index_down.append(own.nearest_ind_nans(sig0_interp[:,i], halocline_center + halocline_interval))
            
            #ignoring different BBL diffusivity, since mean over halocline will be used
            # O2raw_gradient = np.gradient(O2raw_interp[:,i], dz , edge_order = 2)
# =============================================================================
#             if third:
#                 O2_conc_gradient = np.gradient(O2_conc_interp[:,i], dz , edge_order = 2)
#             else:
#                 O2_conc_gradient = np.gradient(O2_interp[:,i], dz , edge_order = 2)
#             
#             if third:
#                 O2_conc_interp_save = O2_conc_interp
#             else:
#                 O2_conc_interp_save = O2_conc_interp
# =============================================================================
            O2_conc_gradient = np.gradient(O2_conc_interp[:,i], dz , edge_order = 2)
    
            O2_conc_gradient_whole[:,i] = O2_conc_gradient
            # flux_O2_shih[:,i] = -turbulent_diffusivity_Shih_interp[:,i] * O2raw_gradient*86400  #TODO: check this value
            flux_O2_shih[:,i] = -turbulent_diffusivity_Shih_interp[:,i] * O2_conc_gradient*86400  #
            #86400 = s/day; converts from m*mumol/(l*s) to mmol/(m^2*d)  ; l to m^3 and micro to milli cancel each other out (factor of 1000)
            
            
            
    # =============================================================================
    #                 flux_O2_over_halocline.append(np.nanmean(flux_O2_shih[halocline_index_down[-1]:halocline_index_up[-1],i]))
    # 
    #                 N2_halocline.append(-9.81*(sig0_interp[halocline_index_up[-1],i]-sig0_interp[halocline_index_down[-1],i])/(z[halocline_index_up[-1]]-z[halocline_index_down[-1]]))
    #                 
    #                 start, end = calcStartAndEndtime(groundpath_mss + finepath_mss_matlab + filename, i)  #very time consuming!!!
    #                 z_N2_up.append(z[halocline_index_up[-1]])
    #                 z_N2_down.append(z[halocline_index_down[-1]])
    #                 casts.append(cast)                
    # =============================================================================
    
        # start, end = calcStartAndEndtime(groundpath_mss + finepath_mss_matlab + filename, 0)  #very time consuming!!!
        # start2, end2 = calcStartAndEndtime(groundpath_mss + finepath_mss_matlab + filename, -1)  #very time consuming!!!
        
        # timedelta = (end2-start)/2
        # times_all.append(start + timedelta)

        np.savez(path_results_npz + filename[:-4] + '_interp', 
                 cast_nr = df_mss['castnumber'], z = z, 
                 lon_start = df_mss['lon_start'], lat_start = df_mss['lat_start'],
                 time_start = df_mss['time_start'], time_end = df_mss['time_end'],          
                 sig0_interp = sig0_interp, N_squared_MIX_interp = N_squared_MIX_interp,
                 eps_MIX_interp = eps_MIX_interp, Reb_MIX_interp = Reb_MIX_interp,
                 O2_conc_gradient = O2_conc_gradient_whole, flux_O2_shih = flux_O2_shih,
                 O2_conc_interp = O2_conc_interp)    
        
        cut_tr1_8 = 47
        if (third == True) and (filename == 'TR1-8.mat'):

            np.savez(path_results_npz + 'TR1-8_1_interp', 
                     cast_nr = df_mss['castnumber'][0:cut_tr1_8], z = z, 
                     lon_start = df_mss['lon_start'][0:cut_tr1_8], lat_start = df_mss['lat_start'][0:cut_tr1_8],
                     time_start = df_mss['time_start'][0:cut_tr1_8], time_end = df_mss['time_end'][0:cut_tr1_8],          
                     sig0_interp = sig0_interp[:,0:cut_tr1_8], N_squared_MIX_interp = N_squared_MIX_interp[:,0:cut_tr1_8],
                     eps_MIX_interp = eps_MIX_interp[:,0:cut_tr1_8], Reb_MIX_interp = Reb_MIX_interp[:,0:cut_tr1_8],
                     O2_conc_gradient = O2_conc_gradient_whole[:,0:cut_tr1_8], flux_O2_shih = flux_O2_shih[:,0:cut_tr1_8],
                     O2_conc_interp = O2_conc_interp[:,0:cut_tr1_8])
            
            np.savez(path_results_npz + 'TR1-8_2_interp', 
                     cast_nr = df_mss['castnumber'][cut_tr1_8:], z = z, 
                     lon_start = df_mss['lon_start'][cut_tr1_8:], lat_start = df_mss['lat_start'][cut_tr1_8:],
                     time_start = df_mss['time_start'][cut_tr1_8:], time_end = df_mss['time_end'][cut_tr1_8:],          
                     sig0_interp = sig0_interp[:,cut_tr1_8:], N_squared_MIX_interp = N_squared_MIX_interp[:,cut_tr1_8:],
                     eps_MIX_interp = eps_MIX_interp[:,cut_tr1_8:], Reb_MIX_interp = Reb_MIX_interp[:,cut_tr1_8:],
                     O2_conc_gradient = O2_conc_gradient_whole[:,cut_tr1_8:], flux_O2_shih = flux_O2_shih[:,cut_tr1_8:],
                     O2_conc_interp = O2_conc_interp[:,0:cut_tr1_8])  
        
    # =============================================================================
    #     #plot
    # =============================================================================
        #O2 flux
        
# =============================================================================
#         fig = pl.figure(1, figsize =(12,6))
#         pl.clf()
#         # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
#         # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
#         pl.pcolormesh(df_mss['lon_start'],  z , flux_O2_shih, vmin = -10, vmax = 10,cmap=cmocean.cm.balance)
#     
#         col = pl.colorbar()
#         col.set_label(r'$F_{O2} [mmol \ m^{-2}d^{-1}]$')
#     
#         
#         pl.plot(df_mss['lon_start'], z[halocline_index_up] , color = 'tab:blue', label = 'halocline depths')
#         pl.plot(df_mss['lon_start'], z[halocline_index_down] , color = 'tab:blue')
#         # pl.scatter(df_mss['lon_start'], valid , color = 'tab:red', label = 'valid')
#     
#         CS = pl.contour(df_mss['lon_start'], z, sig0_interp, 15, colors = 'k')
#         pl.clabel(CS, inline = 1)
#         # custom_line = [Line2D([0], [0], color = 'k', lw = 2)]
#         # CS.collections[1].set_label()
#         pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
#         pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
#         pl.scatter(meta[year]['lon_deep'], -meta[year]['depth_deep'] + meta[year]['highest_sensor_deep'], color = 'tab:orange', label = 'highest sensor')
#         pl.scatter(meta[year]['lon_shallow'], -meta[year]['depth_shallow'] + meta[year]['highest_sensor_shallow'], color = 'tab:orange')
#         
#         # pl.xlim([20.58, 20.64])
#         pl.xlabel('longitude at cast start')
#         pl.ylabel('depth[m]')
#         
#         border_left_3km = 20.6
#         border_right_3km = 20.64996726
#         border_left_1km = 20.59 + 0.01
#         border_right_1km = 20.60665576 + 0.01
#         distance3km = own.haversine_distance(57.32, border_left_3km, 57.32, border_right_3km)    #3km
#         distance1km = own.haversine_distance(57.32, border_left_1km, 57.32, border_right_1km)     #1km
#     
#         scalebar_vertical = -120
#         if first or second or third == True:
#             pl.hlines(scalebar_vertical, border_left_1km, border_right_1km, color = 'tab:grey')
#             pl.vlines(border_left_1km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#             pl.vlines(border_right_1km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#             ax = pl.gca()
#             ax.text((border_left_1km + border_right_1km)/2, scalebar_vertical + 3, '1km', color = 'tab:grey' )
#     
#         else:
#             pl.hlines(scalebar_vertical, border_left_3km, border_right_3km, color = 'tab:grey')
#             pl.vlines(border_left_3km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#             pl.vlines(border_right_3km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#             ax = pl.gca()
#             ax.text((border_left_3km + border_right_3km)/2, scalebar_vertical + 3, '3km', color = 'tab:grey' )
#     
#         # pl.legend(custom_line, [r'$\sigma_0$'], loc='upper right')
#         pl.legend(loc = 'upper left')
#         pl.tight_layout()
#     
#         pl.savefig(path_results_figures + cruise + '_' + filename[:-4] + 'flux_O2.png', dpi = 300)
# =============================================================================
        
    # =============================================================================
        #sig0
        
# =============================================================================
#         fig = pl.figure(1, figsize =(12,6))
#         pl.clf()
#         # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
#         # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
#         pl.pcolormesh(df_mss['lon_start'],  z , sig0_interp, cmap=cmocean.cm.dense)
#     
#         col = pl.colorbar()
#         col.set_label('sig0')
#         pl.xlabel('longitude at cast start')
#         pl.ylabel('depth[m]')
#         pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
#         pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
#         pl.legend(loc = 'upper left')
#         pl.tight_layout()
#     
#         pl.savefig(path_results_figures + cruise + '_' + filename[:-4] + 'sig0.png', dpi = 300)
# =============================================================================
        
        # =============================================================================
        #N2
        
# =============================================================================
#         fig = pl.figure(1, figsize =(12,6))
#         pl.clf()
#         # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
#         # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
#         pl.pcolormesh(df_mss['lon_start'],  z , N_squared_MIX_interp, vmin = 0, vmax = 0.01, cmap=cmocean.cm.dense)
#         col = pl.colorbar()
#         col.set_label('N2')
#         
#         CS = pl.contour(df_mss['lon_start'], z, sig0_interp, 15, colors = 'k')
#         pl.clabel(CS, inline = 1)
# 
#     
# 
#         pl.xlabel('longitude at cast start')
#         pl.ylabel('depth[m]')
#         pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
#         pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
#         pl.legend(loc = 'upper left')
#         pl.tight_layout()
#     
#         pl.savefig(path_results_figures + cruise + '_' + filename[:-4] + 'N2.png', dpi = 300)
# =============================================================================
        
        # =============================================================================
        #                 #O2
        # =============================================================================
                        
        fig = pl.figure(1, figsize =(12,6))
        pl.clf()
        # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
        # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
        newcmap_oxy = cmocean.tools.crop_by_percent(cmocean.cm.thermal_r, 20, which='min', N=None)
        pl.pcolormesh(df_mss['lon_start'],  z , O2_conc_interp, cmap = cmocean.cm.curl_r, vmin = 0, vmax = 350, rasterized = True)# cmap=newcmap_oxy)
        col = pl.colorbar()
        col.set_label(r'$O_2 [\mu mol \ l^{-1}]$')
        CS = pl.contour(df_mss['lon_start'], z, sig0_interp, 15, colors = 'k')
        pl.clabel(CS, inline = 1)
        
        pl.plot(df_mss['lon_start'], z[halocline_index_up] , color = 'magenta', label = 'halocline depths')
        pl.plot(df_mss['lon_start'], z[halocline_index_down] , color = 'magenta')

        pl.xlabel('longitude at cast start')
        pl.ylabel('depth[m]')
        if first or second:
            pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring position')
        elif third:
            pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
            pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
        
        
        pl.legend(loc = 'upper left')
        pl.tight_layout()
    
        # pl.savefig(path_results_figures + cruise + '_' + filename[:-4] + 'O2.png', dpi = 300)
        pl.savefig(path_results_figures + cruise + '_' + filename[:-4] + 'O2.pdf')
        
        if (third == True) and (filename == 'TR1-8.mat'):
            # =============================================================================
            #     #plots for TR1-8:
            # =============================================================================
            #O2 flux
            borders_low = (0, cut_tr1_8)
            borders_high = (cut_tr1_8, -1)
            names = ['TR1-8_1', 'TR1-8_2']
            for count, border_low in enumerate(borders_low):
            
# =============================================================================
#                 fig = pl.figure(1, figsize =(12,6))
#                 pl.clf()
#                 # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
#                 # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
#                 pl.pcolormesh(df_mss['lon_start'][border_low:borders_high[count]],  z , flux_O2_shih[:,border_low:borders_high[count]], vmin = -10, vmax = 10,cmap=cmocean.cm.balance)
#             
#                 col = pl.colorbar()
#                 col.set_label(r'$F_{O2} [mmol \ m^{-2}d^{-1}]$')
#             
#                 
#                 pl.plot(df_mss['lon_start'][border_low:borders_high[count]], z[halocline_index_up][border_low:borders_high[count]] , color = 'tab:blue', label = 'halocline depths')
#                 pl.plot(df_mss['lon_start'][border_low:borders_high[count]], z[halocline_index_down][border_low:borders_high[count]] , color = 'tab:blue')
#                 # pl.scatter(df_mss['lon_start'], valid , color = 'tab:red', label = 'valid')
#             
#                 CS = pl.contour(df_mss['lon_start'][border_low:borders_high[count]], z, sig0_interp[:,border_low:borders_high[count]], 15, colors = 'k')
#                 pl.clabel(CS, inline = 1)
#                 # custom_line = [Line2D([0], [0], color = 'k', lw = 2)]
#                 # CS.collections[1].set_label()
#                 pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
#                 pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
#                 pl.scatter(meta[year]['lon_deep'], -meta[year]['depth_deep'] + meta[year]['highest_sensor_deep'], color = 'tab:orange', label = 'highest sensor')
#                 pl.scatter(meta[year]['lon_shallow'], -meta[year]['depth_shallow'] + meta[year]['highest_sensor_shallow'], color = 'tab:orange')
#                 
#                 # pl.xlim([20.58, 20.64])
#                 pl.xlabel('longitude at cast start')
#                 pl.ylabel('depth[m]')
#                 
#                 border_left_3km = 20.6
#                 border_right_3km = 20.64996726
#                 border_left_1km = 20.59 + 0.01
#                 border_right_1km = 20.60665576 + 0.01
#                 distance3km = own.haversine_distance(57.32, border_left_3km, 57.32, border_right_3km)    #3km
#                 distance1km = own.haversine_distance(57.32, border_left_1km, 57.32, border_right_1km)     #1km
#             
#                 scalebar_vertical = -120
#                 if first or second or third == True:
#                     pl.hlines(scalebar_vertical, border_left_1km, border_right_1km, color = 'tab:grey')
#                     pl.vlines(border_left_1km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#                     pl.vlines(border_right_1km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#                     ax = pl.gca()
#                     ax.text((border_left_1km + border_right_1km)/2, scalebar_vertical + 3, '1km', color = 'tab:grey' )
#             
#                 else:
#                     pl.hlines(scalebar_vertical, border_left_3km, border_right_3km, color = 'tab:grey')
#                     pl.vlines(border_left_3km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#                     pl.vlines(border_right_3km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
#                     ax = pl.gca()
#                     ax.text((border_left_3km + border_right_3km)/2, scalebar_vertical + 3, '3km', color = 'tab:grey' )
#             
#                 # pl.legend(custom_line, [r'$\sigma_0$'], loc='upper right')
#                 pl.legend(loc = 'upper left')
#                 pl.tight_layout()
#             
#                 pl.savefig(path_results_figures + cruise + '_' + names[count] + 'flux_O2.png', dpi = 300)
# =============================================================================
                
            # =============================================================================
# =============================================================================
#                 #sig0
# =============================================================================
                
# =============================================================================
#                 fig = pl.figure(1, figsize =(12,6))
#                 pl.clf()
#                 # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
#                 # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
#                 pl.pcolormesh(df_mss['lon_start'][border_low:borders_high[count]],  z , sig0_interp[:,border_low:borders_high[count]], cmap=cmocean.cm.dense)
#             
#                 col = pl.colorbar()
#                 col.set_label('sig0')
#                 CS = pl.contour(df_mss['lon_start'][border_low:borders_high[count]], z, sig0_interp[:,border_low:borders_high[count]], 15, colors = 'k')
#                 pl.clabel(CS, inline = 1)
# 
#                 
#                 pl.xlabel('longitude at cast start')
#                 pl.ylabel('depth[m]')
#                 pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
#                 pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
#                 pl.legend(loc = 'upper left')
#                 pl.tight_layout()
#             
#                 pl.savefig(path_results_figures + cruise + '_' + names[count] + 'sig0.png', dpi = 300)
# =============================================================================
                
                # =============================================================================
# =============================================================================
#                 #N2
# =============================================================================
                
# =============================================================================
#                 fig = pl.figure(1, figsize =(12,6))
#                 pl.clf()
#                 # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
#                 # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
#                 pl.pcolormesh(df_mss['lon_start'][border_low:borders_high[count]],  z , N_squared_MIX_interp[:,border_low:borders_high[count]], vmin = 0, vmax = 0.01, cmap=cmocean.cm.dense)
#             
#                 col = pl.colorbar()
#                 col.set_label('N2')
#                 CS = pl.contour(df_mss['lon_start'][border_low:borders_high[count]], z, sig0_interp[:,border_low:borders_high[count]], 15, colors = 'k')
#                 pl.clabel(CS, inline = 1)
# 
#                 
#                 pl.xlabel('longitude at cast start')
#                 pl.ylabel('depth[m]')
#                 pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
#                 pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
#                 pl.legend(loc = 'upper left')
#                 pl.tight_layout()
#             
#                 pl.savefig(path_results_figures + cruise + '_' + names[count] + 'N2.png', dpi = 300)
# =============================================================================
                
# =============================================================================
#                 #O2
# =============================================================================
                
                fig = pl.figure(1, figsize =(12,6))
                pl.clf()
                # pl.title(finepath_mss_matlab + filename[:-4] + ' --- start: ' + str(start)[0:19] + '; end: ' + str(end2)[0:19])
                # pl.pcolormesh(df_mss['lon_start'], z, O2_interp, vmin = 0, vmax = 110)
                pl.pcolormesh(df_mss['lon_start'][border_low:borders_high[count]],  z , O2_conc_interp[:,border_low:borders_high[count]], cmap=cmocean.cm.curl_r, vmin = 0, vmax = 350, rasterized = True)#, vmin = 0, vmax = 0.01)
            
                col = pl.colorbar()
                col.set_label('O2')
                CS = pl.contour(df_mss['lon_start'][border_low:borders_high[count]], z, sig0_interp[:,border_low:borders_high[count]], 15, colors = 'k')
                pl.clabel(CS, inline = 1)

                
                pl.xlabel('longitude at cast start')
                pl.ylabel('depth[m]')
                pl.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
                pl.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
                pl.legend(loc = 'upper left')
                pl.tight_layout()
            
                pl.savefig(path_results_figures + cruise + '_' + names[count] + 'O2.pdf')
                # pl.savefig(path_results_figures + cruise + '_' + names[count] + 'O2.png', dpi = 300)

    
starttime = timing.time()
# cruise_years = ['2018']
cruise_years = ['2017', '2018' , '2019']
for cruise_year in cruise_years:
    if cruise_year == '2017':
        main(True, False, False, True)
        # main(True, False, False, False)
    elif cruise_year == '2018':
        main(False, True, False, True)
        # main(False, True, False, False)
    elif cruise_year == '2019':
        main(False, False, True, True)
        # main(False, False, True, False)

endtime = timing.time()
print(endtime - starttime)
    