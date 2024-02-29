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

#plot params
MINI_SIZE = 18
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
pl.rc('font', size=SMALL_SIZE)          # controls default text sizes
pl.rc('axes', titlesize=SMALL_SIZE, titleweight = "bold")     # fontsize of the axes title
pl.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
pl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
pl.rc('legend', fontsize=MINI_SIZE)    # legend fontsize
pl.rc('figure', titlesize=MEDIUM_SIZE, titleweight = "bold")  # fontsize of the figure title

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
# # zoomed = True
# zoomed = False
# filtered = 0
# deep = True
# # deep = False
# =============================================================================


def main(first, second, third, deep, filtered = 0):
    print('currently processing: ', first, second, third, deep, filtered)
    
    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    year, year_number = own.get_year_metadata(first, second, third)
    lon_mooring, lat_mooring = own.get_mooring_locations(first, second, third, deep)
    # operating_system = 'windoof'
    
    g = -9.81
    Omega = 7.2921e-5
    f = 2 * Omega * np.sin(meta[year]['lat_deep']*np.pi/180)
    print(f)
    rho0 = 1000
    prefactor_grad = -g/(rho0*f)
    
    if first == True:
        transect_list = ['TS17.mat', 'TS18.mat', 'TS19.mat', 'TS110.mat', 'TS111.mat', 'TS112.mat', 'TS113.mat', 'TS114.mat', 'TS115.mat', 'TS116.mat']
        N_files = 10
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
                         'TS1_10.mat', 'TS1_11.mat', 'TS1_12.mat', 'TS1_13.mat', 'TS1_14.mat', 'TS1_15.mat']#, 'TS1_16.mat']
        fs_up = md.z18.adcp_fs_up
        fs_down = md.z18.adcp_fs_down
        N_files = 15
    
        halocline_center = md.z18.halocline_center
        halocline_interval = md.z18.halocline_interval
        # transect_list =  ['TS1_5.mat']
        
    elif third == True:
        if deep == True:
            transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8_2.mat', 'TR1-9.mat',
                         'TR1-10.mat', 'TR1-11.mat', 'TR1-12.mat', 'TR1-13.mat', 'TR1-14.mat'#, 'S106-1.mat', 'S106-2.mat',
                         #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                         ]
            N_files = 14
        else: 
            transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8_2.mat', 'TR1-9.mat',
                                 'TR1-10.mat'#, 'TR1-11.mat', 'S106-1.mat', 'S106-2.mat',
                                 #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                                 ]
            N_files = 10
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
    
    adcp_data_low = np.load(path_ADCP_data + 'npz/adcp_low_freqs.npz', allow_pickle = True)
    
    downward_adcp = False
    try:
        adcp_data_low_down = np.load(path_ADCP_data + 'npz/adcp_low_freqs_down.npz', allow_pickle = True)
        downward_adcp = True
    except:
        pass
        
    
    adcp_data_high = np.load(path_ADCP_data + 'npz/adcp_high_freqs.npz', allow_pickle = True)
    
    # =============================================================================
    t_matlab_adcp_low = adcp_data_low['t_matlab']
    t_adcp_low = adcp_data_low['t']
    # print(t_adcp[0:6])
    depth_adcp_low = adcp_data_low['depth'] 
    mdepth_adcp_low= adcp_data_low['mdepth']
    p_adcp_low = adcp_data_low['p']
    currx_adcp_low = adcp_data_low['currx']
    curry_adcp_low = adcp_data_low['curry']
    
    shear2_adcp_low = own.calcShear(currx_adcp_low, curry_adcp_low, depth_adcp_low)
    
    if downward_adcp:
        t_adcp_low_down = adcp_data_low_down['t']
        # print(t_adcp[0:6])
        depth_adcp_low_down = adcp_data_low_down['depth'] 
        curry_adcp_low_down = adcp_data_low_down['curry']
    
    t_matlab_adcp_high = adcp_data_high['t_matlab']
    t_adcp_high = adcp_data_high['t']
    # print(t_adcp[0:6])
    depth_adcp_high = adcp_data_high['depth'] 
    mdepth_adcp_high= adcp_data_high['mdepth']
    p_adcp_high = adcp_data_high['p']
    currx_adcp_high = adcp_data_high['currx']
    curry_adcp_high = adcp_data_high['curry']
    
    shear2_adcp_high = own.calcShear(currx_adcp_high, curry_adcp_high, depth_adcp_high)
    
    t_wind, u_wind, v_wind = own.get_wind_model_data(path_to_wind, first, second, third)
    
    offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh = own.getADCP_plot_definitions(first, second, third, deep)
    largeTicks = mdates.DayLocator()
    smallTicks = mdates.HourLocator(byhour=[6, 12, 18])
    dayformat = mdates.DateFormatter('%d.%b')
    
    depth_adcp_low = depth_adcp_low + offset_adcp
    if downward_adcp:
        depth_adcp_low_down = depth_adcp_low_down + offset_adcp_down
        
    # =============================================================================
    #     #for MSS data
    # =============================================================================
    #for storage of casts closest to deep mooring
    d_nearest = {'filename': [], 'castnumber': [], 'lat_start': [], 'lon_start': [],
                 'time_start': [], 'time_end': []}
    df_nearest = pd.DataFrame(data=d_nearest)
    
    # path_results_figures = 'D:/olivert/results/allCruises/mss_transects/figures/' + cruise + '/'
    path_data_npz = path_mss_interpolated + cruise + '/npz/'
    # path_results_npz = path_results_temp + 'npz/'
    path_results_temp = path_results
    # path_results_figures = path_results_temp + 'figures/'
    
    mss_data = np.load(path_data_npz + transect_list[0][:-4] + '_interp.npz', allow_pickle = True)
    z = mss_data['z']
    
    ind_start = 55
    ind_start = 320
    O2_nearest = np.zeros((len(z), N_files)) # z is the depth, N_files is the number of files
    sig0_nearest = np.zeros((len(z), N_files)) # z is the depth, N_files is the number of files
    v_from_sig0_nearest = np.zeros((len(z)-ind_start, N_files)) # z is the depth, N_files is the number of files
    filecounter = 0
    
    distance_arr = []
    
    # for filename in transect_list:        
        
    for transect in transect_list:
    
        closest_cast = 0
        
        mss_data = np.load(path_data_npz + transect[:-4] + '_interp.npz', allow_pickle = True)
    # =============================================================================
    #         keys:                cast_nr, z, lon_start, lat_start, time_start, time_end,        
    #                              sig0_interp, N_squared_MIX_interp, eps_MIX_interp, Reb_MIX_interp,
    #                              O2_conc_gradient, flux_O2_shih
    # =============================================================================
    
        lons = np.zeros(len(mss_data['lon_start']))
        centered_lons = np.zeros(len(mss_data['lon_start']))
        diff_lons = np.diff(mss_data['lon_start'])
        if diff_lons[0]>0:     
            for count, lon in enumerate(mss_data['lon_start']):
                if deep:
                    lons[count] = own.haversine_distance(meta[year]['lat_deep'], lon, meta[year]['lat_deep'], mss_data['lon_start'][0])
                else:    
                    lons[count] = own.haversine_distance(meta[year]['lat_shallow'], lon, meta[year]['lat_shallow'], mss_data['lon_start'][0])
    
                if count < len(mss_data['lon_start'])-1:
                    centered_lons[count] = (lon + mss_data['lon_start'][count+1])/2
                else:
                    centered_lons[count] = lon + np.mean(diff_lons)/2
        else:
            for count, lon in enumerate(mss_data['lon_start']):
                if deep:
                    lons[count] = -own.haversine_distance(meta[year]['lat_deep'], lon, meta[year]['lat_deep'], mss_data['lon_start'][0])
                else:
                    lons[count] = -own.haversine_distance(meta[year]['lat_shallow'], lon, meta[year]['lat_shallow'], mss_data['lon_start'][0])
    
                if count < len(mss_data['lon_start'])-1:
                    centered_lons[count] = (lon + mss_data['lon_start'][count+1])/2
                else:
                    centered_lons[count] = lon + np.mean(diff_lons)/2
    
        sig0_grad = np.gradient(mss_data['sig0_interp'], lons, axis = 1)
        
        if lons[1] < lons[0]:       #if ship direction is in negative lon dirdf_all = ection
            sig0_grad = -sig0_grad
        else:
            sig0_grad = -sig0_grad  #do it for all, why? casts with increasing lon look also wrong/inverted...; e.g. casts 11/12
    
        
        sig0_grad = np.flip(sig0_grad, axis = 0)        #flip in z direction to fit z array for downward integration
        
        z_grad = mss_data['z']
        z_grad = np.flip(z_grad)
        # print(mss_data['sig0_interp'])
        # sig0_grad = np.gradient(sig0, z, axis = 0)
        #geostrophic balance:
        dv_dz = prefactor_grad * sig0_grad#[1]
        # print(dv_dz)
        #integrate dv_dz from surface or mixed depth to yield v:
        # z_for_v, v_from_sig0 = own.calc_cumsum(dv_dz, z_grad, start = ind_start) #40
        z_for_v, v_from_sig0 = own.calc_cumtrapz(dv_dz, z_grad, start = ind_start) #40  #start/2 = m at current z spacing
    # -mss_data['z']-125
        # print(v_from_sig0)
        # print(-mss_data['z']-125)
        if deep:
            nearest_loc_ind = own.nearest_ind(centered_lons, meta[year]['lon_deep'])
        else:
            nearest_loc_ind = own.nearest_ind(centered_lons, meta[year]['lon_shallow'])
    
        # print(nearest_loc_ind)
    
        mean_adcp_current = np.mean(curry_adcp_low)
        v_bg = mean_adcp_current
        start_nearest = mss_data['time_start'][nearest_loc_ind]
        end_nearest = mss_data['time_end'][nearest_loc_ind]
    
        df_nearest.loc[len(df_nearest.index)] = [transect, nearest_loc_ind, mss_data['lat_start'][nearest_loc_ind], mss_data['lon_start'][nearest_loc_ind], start_nearest, end_nearest]
        # O2_nearest[:,filecounter] = mss_data['O2_interp'][:,nearest_loc_ind]
        sig0_nearest[:,filecounter] = mss_data['sig0_interp'][:,nearest_loc_ind]
        if year_number == 2018:
            
            v_from_sig0_nearest[:,filecounter] = v_from_sig0[:,nearest_loc_ind] + 0
        else:
            v_from_sig0_nearest[:,filecounter] = v_from_sig0[:,nearest_loc_ind] + v_bg
    
    
        cast_starttime = mss_data['time_start'][0]
        cast_endtime = mss_data['time_end'][-1]
        adcp_ind_start = own.nearest_ind_nans(t_adcp_low, df_nearest['time_start'].iloc[filecounter])
        adcp_ind_end = own.nearest_ind_nans(t_adcp_low, df_nearest['time_end'].iloc[filecounter])
        
        #calculate distance vgeo to vadcp:
        distance = 0
        counter = 0
        # print(z_for_v)
        # print(depth_adcp_low)
        for count, value in enumerate(v_from_sig0[:,nearest_loc_ind]):
            adcp_ind_distance = own.nearest_ind_nans(depth_adcp_low, z_for_v[count])
            # print(z_for_v[count], depth_adcp_low[adcp_ind_distance])
            if z_for_v[count] < (np.nanmin(depth_adcp_low) - abs(depth_adcp_low[0] - depth_adcp_low[1])):
                pass
            elif z_for_v[count] > (np.nanmax(depth_adcp_low) + abs(depth_adcp_low[0] - depth_adcp_low[1])):
                pass
            else:
                # print(z_for_v[count], depth_adcp_low[adcp_ind_distance])
                # print(curry_adcp_low[adcp_ind_start, adcp_ind_distance], v_from_sig0[count, nearest_loc_ind] + v_bg)
                # print(abs(depth_adcp_low[0] - depth_adcp_low[1]), z_for_v[count], value, abs(curry_adcp_low[adcp_ind_start, adcp_ind_distance] - v_from_sig0[count, nearest_loc_ind] + v_bg))
                if np.isnan(value):
                    print('nan')
                else:           
                    if second:
                        distance = distance + abs(curry_adcp_low[adcp_ind_start, adcp_ind_distance] - (v_from_sig0[count, nearest_loc_ind]))
                        counter = counter + 1
                    else:
                        distance = distance + abs(curry_adcp_low[adcp_ind_start, adcp_ind_distance] - (v_from_sig0[count, nearest_loc_ind] + v_bg))
                        counter = counter + 1
        # ax_vel.plot(v_from_sig0[:,nearest_loc_ind] + v_bg, z_for_v,
        # curry_adcp_low[adcp_ind_start,:], depth_adcp_low
        
        
        distance_arr.append(distance/counter)
    
    # =============================================================================
    #     #plot figure for v comparison of each closest cast:
    # =============================================================================
        fig_cast, ax_vel = pl.subplots(1,1, figsize = (5,5))
        ax_vel_sigma = ax_vel.twiny()
        color_sigma = 'tab:green'
    
        ax_vel_sigma.plot(mss_data['sig0_interp'][:,nearest_loc_ind], mss_data['z'], ls = '-.', color = color_sigma, label = r'$\sigma_0$')
        ax_vel_sigma.set_xlim([4.8, 9])
        lw_comp = 2
        
        if year_number == 2018:
            # print('2018 fuck yea')
            ax_vel.plot(v_from_sig0[:,nearest_loc_ind] + 0, z_for_v, lw = lw_comp, label = r'$v_{\text{geo}}$')
        else:
            ax_vel.plot(v_from_sig0[:,nearest_loc_ind] + v_bg, z_for_v, lw = lw_comp, label = r'$v_{\text{geo}}$')
            # print(v_bg)
        ax_vel.plot(curry_adcp_low[adcp_ind_start,:], depth_adcp_low, ls = '--', lw = lw_comp, label = r'$v_{\text{SI}}$', color = 'tab:orange')
        if downward_adcp:
            ax_vel.plot(curry_adcp_low_down[adcp_ind_start,:], depth_adcp_low_down, lw = lw_comp, ls = '--', color = 'tab:orange')
    
        
        ax_vel.set_ylim([-90, 0])
        ax_vel.set_xlabel(r'current velocity $[ms^{-1}]$')
        ax_vel.set_ylabel(r'depth [m]')
        
        # ax_vel.text(-0.2, -6, 'sum(dist) = {:4.1f}'.format(distance))
        # ax_vel.set_title('sum(dist) = {:4.2f}'.format(distance/counter*100))
        
        ax_vel_sigma.set_xlabel(r'$\sigma_0 [kg \ m^{-3}]$')
    
        ax_vel.legend(loc = 'upper left')
        ax_vel.grid()
        ax_vel_sigma.legend(loc = 'upper right')
        ax_vel_sigma.tick_params(axis='x', colors=color_sigma)
        ax_vel_sigma.xaxis.label.set_color(color_sigma)
        
        path_figures = path_results + 'plots/v_geo/'
        
        # if not os.path.exists(path_results + 'mss/mss_v_geo/' + str(cruise) + '/') == True:
        #     os.makedirs(path_results + 'mss/mss_v_geo/' + str(cruise) + '/')
            
        if not os.path.exists(path_results + 'plots/v_geo/' + str(cruise) + '/') == True:
            os.makedirs(path_results + 'plots/v_geo/' + str(cruise) + '/')
        
        # pl.savefig(path_results + 'mss/mss_v_geo/' + str(cruise) + '/' + '{}_{}_v_geo_currents_compare_{}.png'.format(cruise, mooring, transect[:-4]), dpi = 300, bbox_inches='tight')
        pl.savefig(path_figures + str(cruise) + '/' + '{}_{}_v_geo_currents_compare_{}.pdf'.format(cruise, mooring, transect[:-4]), bbox_inches='tight')
    
        fig_cast.clf()
        pl.clf()
        
        
    # =============================================================================
    #     #plot mss transect:
    # =============================================================================
        fig_transect, ax_mss = pl.subplots(1,1, figsize = (9,7))
        pl.title(str(cruise) + str(transect) + ' --- start: ' + str(cast_starttime)[0:19] + '; end: ' + str(cast_endtime)[0:19] + '; v_bg = ' + str(round(v_bg, 3)))
        if year_number == 2018:
            mss_colorplot = ax_mss.pcolormesh(mss_data['lon_start'], z_for_v, v_from_sig0 + 0, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
        else:
            mss_colorplot = ax_mss.pcolormesh(mss_data['lon_start'], z_for_v, v_from_sig0 + v_bg, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
    
        # pl.pcolormesh(centered_lons, z_for_v, v_from_sig0, cmap=pl.cm.RdBu_r, vmin = -0.50, vmax = 0.50)
    
        col = pl.colorbar(mss_colorplot)
        col.set_label(r'$v$ [m/s] from $\sigma_0$')
        # v_colbar_low = pl.colorbar(v_col_low, pad = -0.08,  ax = (ax_wind, ax_v_low, ax_v_geo), shrink = 0.7650, anchor = (1,0), aspect = 15)
        # v_colbar_low.set_label('current speed [m/s]')
    
        CS = ax_mss.contour(mss_data['lon_start'], mss_data['z'], mss_data['sig0_interp'], 15, colors = 'k')
        pl.clabel(CS, inline = 1)
        # custom_line = [Line2D([0], [0], color = 'k', lw = 2)]
        # CS.collections[1].set_label()
        ax_mss.vlines(meta[year]['lon_deep'], -meta[year]['depth_deep'], -meta[year]['depth_deep'] + meta[year]['rope_length_deep'], color = 'tab:orange', label = 'mooring positions')
        ax_mss.vlines(meta[year]['lon_shallow'], -meta[year]['depth_shallow'], -meta[year]['depth_shallow'] + meta[year]['rope_length_shallow'], color = 'tab:orange')
        ax_mss.scatter(meta[year]['lon_deep'], -meta[year]['depth_deep'] + meta[year]['highest_sensor_deep'], color = 'tab:orange', label = 'highest sensor')
        ax_mss.scatter(meta[year]['lon_shallow'], -meta[year]['depth_shallow'] + meta[year]['highest_sensor_shallow'], color = 'tab:orange')
    
        # pl.xlim([20.58, 20.64])
        pl.xlabel('longitude at cast start')
        pl.ylabel('depth[m]')
        
        border_left_3km = 20.6
        border_right_3km = 20.64996726
        border_left_1km = 20.59 + 0.01
        border_right_1km = 20.60665576 + 0.01
        distance3km = own.haversine_distance(57.32, border_left_3km, 57.32, border_right_3km)    #3km
        distance1km = own.haversine_distance(57.32, border_left_1km, 57.32, border_right_1km)     #1km
    # =============================================================================
    #     if zoomed == True:
    #         scalebar_vertical = -100
    #     else: 
    #         scalebar_vertical = -120
    #     if first or second or third == True:
    #         pl.hlines(scalebar_vertical, border_left_1km, border_right_1km, color = 'tab:grey')
    #         pl.vlines(border_left_1km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
    #         pl.vlines(border_right_1km, scalebar_vertical-1, scalebar_vertical + 1, color = 'tab:grey')
    #         ax = pl.gca()
    #         ax.text((border_left_1km + border_right_1km)/2, scalebar_vertical + 3, '1km', color = 'tab:grey' )
    # =============================================================================
    
        # fig_transect.savefig(path_results + 'mss/mss_v_geo/' + str(cruise) + '/' + str(cruise) + str(mooring) + '_v_from_sig0_{}.png'.format(transect[:-4]), dpi = 300, bbox_inches='tight')
        fig_transect.savefig(path_figures + str(cruise) + '/' + str(cruise) + str(mooring) + '_v_from_sig0_{}.pdf'.format(transect[:-4]), bbox_inches='tight')
    
        fig_transect.clf()
        pl.clf()
        
        filecounter += 1
     
    # =============================================================================
    # #for comparison plot
    # =============================================================================
    #plot params
    MINI_SIZE = 11
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    pl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    pl.rc('axes', titlesize=SMALL_SIZE, titleweight = "bold")     # fontsize of the axes title
    pl.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    pl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    pl.rc('legend', fontsize=MINI_SIZE)    # legend fontsize
    pl.rc('figure', titlesize=MEDIUM_SIZE, titleweight = "bold")  # fontsize of the figure title
    
    #build transparent dummy hours
    cruise_start = df_nearest['time_start'][0] - np.timedelta64(24, 'h')
    cruise_end = df_nearest['time_end'].iloc[-1] + np.timedelta64(48, 'h')
    # dt = df_nearest['time_end'].iloc[-1]-cruise_start
    # hours = (dt.days + 2)*24
    cruise_start_hours = cruise_start.replace(minute=0, second=0, microsecond=0)
    cruise_end_hours = cruise_end.replace(minute=0, second=0, microsecond=0)
    dummy_times = np.arange(cruise_start_hours, cruise_end_hours, dtype = 'datetime64[h]')
    dummy_v_geo = np.empty((len(z)-ind_start, len(dummy_times),) )
    dummy_v_geo[:] = np.nan
    
    #width of calculated geostrophic time plots, in each direction around center
    if third:
        dummy_dt = np.timedelta64(3, 'h')
    elif second:
        dummy_dt = np.timedelta64(2, 'h')
    else:
        dummy_dt = np.timedelta64(2, 'h')
    
    
    for count, time in enumerate(dummy_times):
        for transect_nr in range(len(df_nearest)):
            if (time > (df_nearest['time_start'].iloc[transect_nr] - dummy_dt)) and (time < (df_nearest['time_start'].iloc[transect_nr] + dummy_dt)):
                dummy_v_geo[:, count] = v_from_sig0_nearest[:, transect_nr]
            
    if deep:
        ylim1 = -90
        ylim2 = -30
    else:
        ylim1 = -77
        ylim2 = -30
    
    
        #########################################################
    plot_wind = True
    plot_wind = False
    
    
    if plot_wind:
        gridspec_all = {'height_ratios': [1,2,2]}
        size_fig = (10,8)
        fig, (ax_wind, ax_v_low, ax_v_geo) = pl.subplots(3,1, sharex=True, gridspec_kw=gridspec_all, figsize = size_fig)
        ax_names = (ax_wind, ax_v_low, ax_v_geo)
    else:
        gridspec_all = {'height_ratios': [1,1]}
        size_fig = (10,6)
        fig, (ax_v_low, ax_v_geo) = pl.subplots(2,1, sharex=True, gridspec_kw=gridspec_all, figsize = size_fig)
        ax_names = (ax_v_low, ax_v_geo)
    
    if plot_wind:
        ax_wind.quiver(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2), np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.005, scale = 50)#, width = 0.009)#width 0.004, scale 40
            # pl.quiver(t_wind, 0, u_wind[:], v_wind[:], width=.002)
        # ax_wind.quiver(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2), np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid')#, width = 0.4, scale = 10)#width 0.004, scale 40
    
        ax_wind.set_ylim(0, 12)
        ax_wind.set_ylabel('speed [m/s]')
        ax_wind.grid()
        # own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
        ax_wind.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    #v low
    
    v_col_low = ax_v_low.pcolormesh(adcp_data_low['t'], depth_adcp_low, curry_adcp_low.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
    if downward_adcp:
        v_col_low_down = ax_v_low.pcolormesh(adcp_data_low_down['t'], depth_adcp_low_down, curry_adcp_low_down.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
    
    
    if plot_wind:
        v_colbar_low = pl.colorbar(v_col_low, pad = -0.08,  ax = (ax_wind, ax_v_low, ax_v_geo), shrink = 0.7650, anchor = (1,0), aspect = 15)
    else:
        v_colbar_low = pl.colorbar(v_col_low, pad = -0.08,  ax = (ax_v_low, ax_v_geo), shrink = 1, anchor = (1,0), aspect = 15)
    v_colbar_low.set_label(r'$v \ [m/s]$')
    ax_v_low.set_ylim([ylow, yhigh])
    ax_v_low.set_ylabel('depth [m]')
    #plot closest cast 
    ax_v_low.scatter(df_nearest['time_start'], np.full_like(df_nearest['time_start'], -30, dtype = int), color = 'black', marker = 'v', s = 100)
    # ax_v_low.scatter(df_nearest['time_end'], np.full_like(df_nearest['time_end'], -30, dtype = int), color = 'black', marker = 'v')
    
            # ax_adcp.scatter(time_start, -1*np.ones(len(time_start)),marker = 'v', color = 'black')
    if plot_wind:
        for count, transect_name in enumerate(transect_list):
            ax_v_low.text(df_nearest['time_start'][count], -49, transect_name[:-4], 
             color = 'black', rotation = 90, 
             rotation_mode = 'anchor')
    else:
        for count, transect_name in enumerate(transect_list):
            ax_v_low.text(df_nearest['time_start'][count], -29, transect_name[:-4], 
             color = 'black', rotation = 45, 
             rotation_mode = 'anchor')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_v_low.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    
    # fig = pl.figure(1, figsize =(12,6))
    # pl.clf()
    # pl.title('at lon: ' + str(df_nearest['lon_start']) + '; buoy deep at lon: ' + str(meta[year]['lon_deep']))
    # ax_v_geo.pcolormesh(df_nearest['time_start'], z_for_v, v_from_sig0_nearest + v_bg, cmap=cmocean.cm.balance, vmin = -0.25, vmax = 0.25)
    ax_v_geo.pcolormesh(dummy_times, z_for_v, dummy_v_geo, cmap=cmocean.cm.balance, vmin = -0.25, vmax = 0.25, rasterized = True)
    ax_v_geo.set_ylabel('depth [m]')
    
    # =============================================================================
    # for count, distance in enumerate(distance_arr):
    #     ax_v_geo.text(df_nearest['time_start'][count], -50, 'd = {:4.2f}'.format(distance*100), 
    #      color = 'black', rotation = 90, 
    #      rotation_mode = 'anchor')
    # 
    # =============================================================================
    
    # pl.pcolormesh(v_from_sig0_nearest, cmap=cmocean.cm.balance, vmin = -0.25, vmax = 0.25)
    
    # pl.pcolormesh(df_nearest['time_start'], z, sig0_nearest)
    # col = pl.colorbar()
    # col.set_label(r'$v$ [m/s] from $\sigma_0$')
    ax_v_geo.set_ylim([ylim1, ylim2])
    ax_v_low.set_ylim([ylim1, ylim2])
    
    if plot_wind:
        # pass
        for count, ax_name in enumerate(ax_names):
            if first:
                own.plotLetter(ax_name, 0.01, 0.83, count, ha = 'left')
            elif third:
                own.plotLetter(ax_name, 0.01, 0.05, count, ha = 'left')
            else: 
                own.plotLetter(ax_name, 1, 0.2, count)
    else:
        for count, ax_name in enumerate(ax_names):
            if third:
                # own.plotLetter(ax_name, -0.05, 0.04, count)
                own.plotLetter(ax_name, 0.01, 0.83, count, ha = 'left')
            elif second:
                own.plotLetter(ax_name, 1, 0.04, count)
            else:
                own.plotLetter(ax_name, 0.01, 0.83, count, ha = 'left')
    
    if plot_wind:
        # pl.savefig(path_results + 'plots/v_geo/' + '{}_{}_v_from_sig0_time_comparison_wind.png'.format(cruise, mooring), dpi = 300, bbox_inches='tight')
        # pl.savefig(path_results + 'mss/mss_v_geo/' + str(cruise) + '/' + '{}_{}_v_from_sig0_time_comparison_wind.png'.format(cruise, mooring), dpi = 300, bbox_inches='tight')
        pl.savefig(path_figures + '{}_{}_v_from_sig0_time_comparison_wind.pdf'.format(cruise, mooring), bbox_inches='tight')
        pl.savefig(path_figures + str(cruise) + '/' + '{}_{}_v_from_sig0_time_comparison_wind.pdf'.format(cruise, mooring), bbox_inches='tight')
    
    else:
        # pl.savefig(path_results + 'plots/v_geo/' + '{}_{}_v_from_sig0_time_comparison.png'.format(cruise, mooring), dpi = 300, bbox_inches='tight')
        # pl.savefig(path_results + 'mss/mss_v_geo/' + str(cruise) + '/' + '{}_{}_v_from_sig0_time_comparison.png'.format(cruise, mooring), dpi = 300, bbox_inches='tight')
        pl.savefig(path_figures + '{}_{}_v_from_sig0_time_comparison.pdf'.format(cruise, mooring), bbox_inches='tight')
        pl.savefig(path_figures + str(cruise) + '/' + '{}_{}_v_from_sig0_time_comparison.pdf'.format(cruise, mooring), bbox_inches='tight')

    
starttime = timing.time()

# cruise_years = ['2017', '2018']
cruise_years = ['2017', '2018', '2019']

for cruise_year in cruise_years:
    if cruise_year == '2017':
        main(True, False, False, True)

    elif cruise_year == '2018':
        main(False, True, False, True)

    elif cruise_year == '2019':
        main(False, False, True, True)
        # main(False, False, True, False)

endtime = timing.time()
print(endtime - starttime)
