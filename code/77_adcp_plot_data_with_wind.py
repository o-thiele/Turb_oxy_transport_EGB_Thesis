import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import pylab as pl
import os
import matplotlib.dates as mdates
import scipy.io as ssio
from datetime import datetime
import time as timing
import pickle
from scipy import signal
import cmocean
import string
#own functions
import own_functions as own

#plot params
MINI_SIZE = 10
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


def main(first, second, third, deep):
    
    print('currently processing: ', first, second, third, deep)
    
    offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh = own.getADCP_plot_definitions(first, second, third, deep)
    largeTicks = mdates.DayLocator()
    smallTicks = mdates.HourLocator(byhour=[6, 12, 18])
    dayformat = mdates.DateFormatter('%d. %b %y')

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
                         'TS1_10.mat', 'TS1_11.mat', 'TS1_12.mat', 'TS1_13.mat', 'TS1_14.mat', 'TS1_15.mat']#, 'TS1_16.mat']
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
        else: 
            transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8_1.mat', 'TR1-8_2.mat', 'TR1-9.mat',
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
    
    if year == '2019':
        # largeTicks = mdates.DayLocator(interval=2)
        dayformat = mdates.DateFormatter('%d. %b')

    
    path_results_temp = path_results + 'plots/adcp/'
    if not os.path.exists(path_results_temp) == True:
        os.makedirs(path_results_temp)
    # path_results_data = path_ADCP_data + 'npz/'
    # =============================================================================
    # for ADCP data
    # =============================================================================
    
    #interpolated data
    adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated.npz', allow_pickle = True)        
    
    downwards_adcp_data = False
    try:
        adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_interpolated_down_avgd.npz', allow_pickle = True)
        downwards_adcp_data = True
    except:
        print('no downward adcp data found')
        pass
    
    if third:
        try:
            adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated_fine.npz', allow_pickle = True)        
            try:
                adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_interpolated_fine_down.npz', allow_pickle = True)
                downwards_adcp_data = True
            except:
                print('no downward adcp data found')
                pass
        except:
            print('no data with finer cutouts found')
            pass


    #cropped data
# =============================================================================
#     adcp_data = np.load(path_ADCP_data + 'npz/adcp_cropped.npz', allow_pickle = True)
#     
#     downwards_adcp_data = False
#     try:
#         adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_cropped_down.npz', allow_pickle = True)
#         downwards_adcp_data = True
#     except:
#         print('no downward adcp data found')
#         pass
# =============================================================================
    
    # =============================================================================
    # t_matlab_adcp = adcp_data['t_matlab']
    t_adcp = adcp_data['t']
    # print(t_adcp[0:6])
    depth_adcp = adcp_data['depth'] 
    # mdepth_adcp = adcp_data['mdepth']
    # p_adcp = adcp_data['p']
    currx_adcp = adcp_data['currx']
    curry_adcp = adcp_data['curry']
    depth_adcp = depth_adcp + offset_adcp

    
    if downwards_adcp_data:
        # t_matlab_adcp = adcp_data['t_matlab']
        t_adcp_down = adcp_data_down['t']
        # print(t_adcp[0:6])
        depth_adcp_down = adcp_data_down['depth'] 
        # mdepth_adcp_down = adcp_data_down['mdepth']
        # p_adcp_down = adcp_data_down['p']
        currx_adcp_down = adcp_data_down['currx']
        curry_adcp_down = adcp_data_down['curry']
        depth_adcp_down = depth_adcp_down + offset_adcp_down

        
    # =============================================================================

    # =============================================================================
    # #for wind data
    # =============================================================================
    winddata_u = np.genfromtxt(path_to_wind + year + '_u.dump', comments = '%', skip_header=4)
    winddata_v = np.genfromtxt(path_to_wind + year + '_v.dump', comments = '%', skip_header=4)

    tu_wind = winddata_u[0:, 0]
    u_wind = winddata_u[0:, 1]
    v_wind = winddata_v[0:, 1]
    
    t2_wind = np.copy(tu_wind)
    tOffset = -own.secondsUntil(year_number)+own.secondsUntil(1970)           #to convert to epoch time
    
    t2_wind[:] = tu_wind[:] - tOffset
    t_wind = np.arange(0, len(t2_wind)).astype(datetime)
    for count, item in enumerate(t2_wind):
        t_wind[count] = datetime.utcfromtimestamp(item)

# =============================================================================
#     #plots
# =============================================================================
    #u plot
    
    # fig = pl.figure(1, figsize =(12,5))
    # pl.clf()
    # fig.suptitle(cruise + ' ' + mooring + ' --- Shear and other plot', fontsize=14)
    rows = 2
    cols = 1
    gridspec = {'height_ratios': [1,2]}
    fig, (ax_wind, ax) = pl.subplots(rows,cols, sharex=True, gridspec_kw=gridspec, figsize = (7,5))
  
    Y_wind = np.zeros(len(t_wind))
    # ax_wind.quiver(t_wind, 0, u_wind[:], v_wind[:], angles = 'uv', scale_units = 'y', scale = 100, pivot = 'mid')
    # ax_wind.quiver(t_wind, -1, np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.005, scale = 30)
    # pl.scatter(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2))
    ax_wind.quiver(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2), np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.005, scale = 30)
        # pl.quiver(t_wind, 0, u_wind[:], v_wind[:], width=.002)
    ax_wind.set_ylim(0, 15)
    ax_wind.set_ylabel('speed [m/s]')
    ax_wind.grid()
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_wind.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])


    u_col = ax.pcolormesh(adcp_data['t'], depth_adcp, currx_adcp.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25)
    if downwards_adcp_data:
        ax.pcolormesh(adcp_data_down['t'], depth_adcp_down, currx_adcp_down.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25)
    u_colbar = pl.colorbar(u_col, pad = -0.05,  ax = (ax, ax_wind), shrink = 0.60, anchor = (1,0), aspect = 12)

    u_colbar.set_label('u [m/s]')
    ax.set_ylim([ylow, yhigh])
    ax.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])

    fig.savefig(path_results_temp + cruise + mooring + '_wind_u.png', dpi = 300)
    fig.clf()
    
    
    #v plot
    fig, (ax_wind, ax) = pl.subplots(rows,cols, sharex=True, gridspec_kw=gridspec, figsize = (7,5))
  
    Y_wind = np.zeros(len(t_wind))
    # ax_wind.quiver(t_wind, 0, u_wind[:], v_wind[:], angles = 'uv', scale_units = 'y', scale = 100, pivot = 'mid')
    # ax_wind.quiver(t_wind, -1, np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.005, scale = 30)
    # pl.scatter(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2))
    ax_wind.quiver(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2), np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.005, scale = 30)
        # pl.quiver(t_wind, 0, u_wind[:], v_wind[:], width=.002)
    ax_wind.set_ylim(0, 15)
    ax_wind.set_ylabel('speed [m/s]')
    ax_wind.grid()
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_wind.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])


    v_col = ax.pcolormesh(adcp_data['t'], depth_adcp, curry_adcp.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25)
    if downwards_adcp_data:
        ax.pcolormesh(adcp_data_down['t'], depth_adcp_down, curry_adcp_down.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25)
    v_colbar = pl.colorbar(v_col, pad = -0.05,  ax = (ax, ax_wind), shrink = 0.60, anchor = (1,0), aspect = 12)

    v_colbar.set_label('v [m/s]')
    ax.set_ylim([ylow, yhigh])
    ax.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    # ax.legend()
    
    fig.savefig(path_results_temp + cruise + mooring + '_wind_v.png', dpi = 300)
    fig.clf()
    
# =============================================================================
#     #combined uv plot
# =============================================================================
    rows = 3
    cols = 1
    gridspec = {'height_ratios': [1,2,2]}
    fig, (ax_wind, ax, ax_v) = pl.subplots(rows,cols, sharex=True, gridspec_kw=gridspec, figsize = (10,6))
    axs = [ax_wind, ax, ax_v]
  
    Y_wind = np.zeros(len(t_wind))
    # ax_wind.quiver(t_wind, 0, u_wind[:], v_wind[:], angles = 'uv', scale_units = 'y', scale = 100, pivot = 'mid')
    # ax_wind.quiver(t_wind, -1, np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.005, scale = 30)
    # pl.scatter(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2))
    ax_wind.quiver(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2), np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.004, scale = 40)#width 0.004, scale 30
        # pl.quiver(t_wind, 0, u_wind[:], v_wind[:], width=.002)
    ax_wind.set_ylim(0, 15)
    ax_wind.set_ylabel('speed $[ms^{-1}]$')
    ax_wind.grid()
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_wind.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])


    for count, transect in enumerate(transect_list): 
        path_data_npz = path_mss_interpolated + cruise + '/npz/'

        mss_data = np.load(path_data_npz + transect[:-4] + '_interp.npz', allow_pickle = True)
    # =============================================================================
    #         keys:                cast_nr, z, lon_start, lat_start, time_start, time_end,        
    #                              sig0_interp, N_squared_MIX_interp, eps_MIX_interp, Reb_MIX_interp,
    #                              O2_conc_gradient, flux_O2_shih
    # =============================================================================
        ax_wind.scatter(mss_data['time_start'][count], 15, marker = 'v', color = 'black')
        ax_wind.text(mss_data['time_start'][count], 15, transect[:-4], 
         color = 'black', rotation = 70, 
         rotation_mode = 'anchor', fontsize = 9)
    

    u_col = ax.pcolormesh(adcp_data['t'], depth_adcp, currx_adcp.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True, zorder = 3)
    if downwards_adcp_data:
        ax.pcolormesh(adcp_data_down['t'], depth_adcp_down, currx_adcp_down.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True, zorder = 3)
    u_colbar = pl.colorbar(u_col, pad = -0.08,  ax = (ax, ax_v, ax_wind), shrink = 0.7650, anchor = (1,0), aspect = 12)

    u_colbar.set_label('current speed $[ms^{-1}]$')
    ax_v.set_ylim([ylow, yhigh])
    ax_v.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])

    v_col = ax_v.pcolormesh(adcp_data['t'], depth_adcp, curry_adcp.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True, zorder = 3)
    if downwards_adcp_data:
        ax_v.pcolormesh(adcp_data_down['t'], depth_adcp_down, curry_adcp_down.T,cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True, zorder = 3)
    # v_colbar = pl.colorbar(v_col, pad = -0.05,  ax = (ax, ax_v, ax_wind), shrink = 0.60, anchor = (1,0), aspect = 12)

    # v_col.set_label('v [m/s]')
    ax.set_ylim([ylow, yhigh])
    ax.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    ax.grid(zorder = 0)
    ax_v.grid(zorder = 0)

    
    ax_wind.text(0.96, 0.7, string.ascii_lowercase[0]+ ')', transform=ax_wind.transAxes, 
        size=20, weight='bold')
    ax.text(0.96, 0.86, string.ascii_lowercase[1]+ ')', transform=ax.transAxes, 
        size=20, weight='bold')
    ax_v.text(0.96, 0.86, string.ascii_lowercase[2]+ ')', transform=ax_v.transAxes, 
        size=20, weight='bold')
# =============================================================================
#     for n, ax_n in enumerate(axs):
#         #x, y
#         ax_n.text(0.96, 0.75, string.ascii_uppercase[n], transform=ax_n.transAxes, 
#             size=20, weight='bold')
# =============================================================================

    # fig.savefig(path_results_temp + cruise + mooring + '_wind_uv_avgd.png', dpi = 300, bbox_inches='tight')
    fig.savefig(path_results_temp + cruise + mooring + '_wind_uv_avgd.pdf', bbox_inches='tight')

    pl.show()
    fig.clf()

    
starttime = timing.time()

# cruise_years = ['2019']
cruise_years = ['2017', '2018', '2019']
for cruise_year in cruise_years:
    if cruise_year == '2017':
        main(True, False, False, True)
    elif cruise_year == '2018':
        main(False, True, False, True)
    elif cruise_year == '2019':
        main(False, False, True, True)
        main(False, False, True, False)

endtime = timing.time()
print(endtime - starttime)
