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
    
# =============================================================================
# first = True
# first = False
# second = True
# second = False
# third = True
# # third = False
# 
# deep = True
# # deep = False
# =============================================================================

def main(first, second, third, deep):
    offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh = own.getADCP_plot_definitions(first, second, third, deep)
    largeTicks = mdates.DayLocator()
    # if third == True:
    #     largeTicks = mdates.DayLocator(interval=2)
    smallTicks = mdates.HourLocator(byhour=[6, 12, 18])
    dayformat = mdates.DateFormatter('%d. %b %Y')
    if third == True:
        dayformat = mdates.DateFormatter('%d. %b')
    
    
    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    year, year_number = own.get_year_metadata(first, second, third)
    lon_mooring, lat_mooring = own.get_mooring_locations(first, second, third, deep)
    # operating_system = 'windoof'
    
    path_results_temp = path_results + 'plots/adcp/filtered/'
    if not os.path.exists(path_results_temp) == True:
        os.makedirs(path_results_temp)
    
    # path_results_data = path_ADCP_data + 'npz/'
    # =============================================================================
    # for ADCP data
    # =============================================================================
    
    adcp_data_low = np.load(path_ADCP_data + 'npz/adcp_low_freqs.npz', allow_pickle = True)
    
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
    depth_adcp_low = depth_adcp_low + offset_adcp
    
    shear2_adcp_low = own.calcShear(currx_adcp_low, curry_adcp_low, depth_adcp_low)
    
    t_matlab_adcp_high = adcp_data_high['t_matlab']
    t_adcp_high = adcp_data_high['t']
    # print(t_adcp[0:6])
    depth_adcp_high = adcp_data_high['depth'] 
    mdepth_adcp_high= adcp_data_high['mdepth']
    p_adcp_high = adcp_data_high['p']
    currx_adcp_high = adcp_data_high['currx']
    curry_adcp_high = adcp_data_high['curry']
    depth_adcp_high = depth_adcp_high + offset_adcp
    
    shear2_adcp_high = own.calcShear(currx_adcp_high, curry_adcp_high, depth_adcp_high)
    
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
    # #plot
    # =============================================================================
    gridspec_all = {'height_ratios': [2,2,2,2,2,2,2,1,1]}
    size_fig = (10,15)  #old (10,13)
    fig, (ax_wind, ax_u_low, ax_v_low, ax_S_low, ax_u_high, ax_v_high, ax_S_high, ax_col_1, ax_col_2) = pl.subplots(9,1, sharex=True, gridspec_kw=gridspec_all, figsize = size_fig)
    
    #combined uv plot
      
    Y_wind = np.zeros(len(t_wind))
    # ax_wind.quiver(t_wind, 0, u_wind[:], v_wind[:], angles = 'uv', scale_units = 'y', scale = 100, pivot = 'mid')
    # ax_wind.quiver(t_wind, -1, np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.005, scale = 30)
    # pl.scatter(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2))
    ax_wind.quiver(t_wind, np.sqrt(u_wind[:]**2 + v_wind[:]**2), np.full_like(u_wind, 1), np.full_like(u_wind, 1), angles = np.rad2deg(np.arctan2(v_wind[:], u_wind[:])), pivot = 'mid', width = 0.004, scale = 40)#width 0.004, scale 30
        # pl.quiver(t_wind, 0, u_wind[:], v_wind[:], width=.002)
    ax_wind.set_ylim(0, 15)
    ax_wind.set_ylabel(r'wind speed $[ms^{-1}] \ $')
    ax_wind.grid()
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_wind.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    #u low
    u_col_low = ax_u_low.pcolormesh(adcp_data_low['t'], depth_adcp_low, currx_adcp_low.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
    ax_u_low.set_ylim([ylow, yhigh])
    ax_u_low.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_u_low.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    #v low
    v_col_low = ax_v_low.pcolormesh(adcp_data_low['t'], depth_adcp_low, curry_adcp_low.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
    # u_colbar_low = pl.colorbar(u_col_low, pad = -0.08,  ax = (ax_wind, ax_u_low, ax_v_low, ax_u_high, ax_v_high), shrink = 0.7650, anchor = (1,0), aspect = 12)
    # u_colbar_low.set_label('current speed [m/s]')
    ax_v_low.set_ylim([ylow, yhigh])
    ax_v_low.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_v_low.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    #u high
    u_col_high = ax_u_high.pcolormesh(adcp_data_high['t'], depth_adcp_high, currx_adcp_high.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
    # u_colbar_low = pl.colorbar(u_col_low, pad = -0.08,  ax = (ax_wind, ax_u_low, ax_v_low, ax_u_high, ax_v_high), shrink = 0.7650, anchor = (1,0), aspect = 12)
    # u_colbar_low.set_label('current speed [m/s]')
    ax_u_high.set_ylim([ylow, yhigh])
    ax_u_high.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_u_high.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    #v high
    v_col_high = ax_v_high.pcolormesh(adcp_data_high['t'], depth_adcp_high, curry_adcp_high.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
    # u_colbar_low = pl.colorbar(u_col_low, pad = -0.08,  ax = (ax_wind, ax_u_low, ax_v_low, ax_u_high, ax_v_high), shrink = 0.7650, anchor = (1,0), aspect = 12)
    # u_colbar_low.set_label('current speed [m/s]')
    ax_v_high.set_ylim([ylow, yhigh])
    ax_v_high.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_v_high.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    #S2 low
    S2_col_low = ax_S_low.pcolormesh(adcp_data_low['t'], depth_adcp_low, shear2_adcp_low.T, cmap=cmocean.cm.speed, vmin=0,vmax=0.005, rasterized = True)
    ax_S_low.set_ylim([ylow, yhigh])
    ax_S_low.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_S_low.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    
    #S2 high
    S2_col_high = ax_S_high.pcolormesh(adcp_data_high['t'], depth_adcp_high, shear2_adcp_high.T, cmap=cmocean.cm.speed, vmin=0,vmax=0.005, rasterized = True)
    ax_S_high.set_ylim([ylow, yhigh])
    ax_S_high.set_ylabel('depth [m]')
    # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
    own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
    ax_S_high.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
    # ax_S_high.minorticks_on()
    ax_S_high.tick_params(top=False,
                   bottom=True,
                   left=True,
                   right=False,
                   labelleft=True,
                   labelbottom=True)
    # ax_S_high.set_xlabel('time')
    
    ax_col_1.set_axis_off()
    ax_col_2.set_axis_off()
    
    u_colbar_low = pl.colorbar(u_col_low, ax = ax_col_1, orientation = 'horizontal', anchor = (0.5,0), aspect = 100)
    u_colbar_low.set_label(r'current speed $[m/s]$')
    S2_colbar_low = pl.colorbar(S2_col_low,  ax = ax_col_2, orientation = 'horizontal', anchor = (0.5,0), aspect = 100)
    S2_colbar_low.set_label(r'$S^2 [s^{-2}]$')
    
    ax_names = (ax_wind, ax_u_low, ax_v_low, ax_S_low, ax_u_high, ax_v_high, ax_S_high)
    ax_names_no_wind = (ax_u_low, ax_v_low, ax_S_low, ax_u_high, ax_v_high, ax_S_high)
    # def plotLetter(ax_name, x,y,plot_number, size = 20, weight='bold'):
    #     ax_name.text(x, y, string.ascii_uppercase[plot_number], transform=ax_name.transAxes, size=20, weight='bold')
    
    for count, ax_name in enumerate(ax_names):
        own.plotLetter(ax_name, 1, 0.75, count)
    
    
    for count, ax_name in enumerate(ax_names_no_wind):
        ax_name.set_yticks([0,-25,-50,-75])
    # own.plotLetter(ax_wind, 0.96, 0.7, 0)
    # own.plotLetter(ax_u_low, 0.96, 0.84, 1)
    # own.plotLetter(ax_v_low, 0.96, 0.86, 2)
    
    # ax_v_low.text(0.96, 0.86, string.ascii_uppercase[2], transform=ax_v_low.transAxes, 
    #     size=20, weight='bold')
    
    
    # =============================================================================
    # #downward adcp:
    # =============================================================================
    downward_adcp_data = False
    try:
        adcp_data_low = np.load(path_ADCP_data + 'npz/adcp_low_freqs_down_avgd.npz', allow_pickle = True)
        adcp_data_high = np.load(path_ADCP_data + 'npz/adcp_high_freqs_down_avgd.npz', allow_pickle = True)
        downward_adcp_data = True
    except:
        print('no downward adcp data found')
        pass
    
    if downward_adcp_data:
    # =============================================================================
        t_matlab_adcp_low = adcp_data_low['t_matlab']
        t_adcp_low = adcp_data_low['t']
        # print(t_adcp[0:6])
        depth_adcp_low = adcp_data_low['depth'] 
        mdepth_adcp_low= adcp_data_low['mdepth']
        p_adcp_low = adcp_data_low['p']
        currx_adcp_low = adcp_data_low['currx']
        curry_adcp_low = adcp_data_low['curry']
        depth_adcp_low = depth_adcp_low + offset_adcp_down
        
        shear2_adcp_low = own.calcShear(currx_adcp_low, curry_adcp_low, depth_adcp_low)
        
        t_matlab_adcp_high = adcp_data_high['t_matlab']
        t_adcp_high = adcp_data_high['t']
        # print(t_adcp[0:6])
        depth_adcp_high = adcp_data_high['depth'] 
        mdepth_adcp_high= adcp_data_high['mdepth']
        p_adcp_high = adcp_data_high['p']
        currx_adcp_high = adcp_data_high['currx']
        curry_adcp_high = adcp_data_high['curry']
        depth_adcp_high = depth_adcp_high + offset_adcp_down
    
        
        shear2_adcp_high = own.calcShear(currx_adcp_high, curry_adcp_high, depth_adcp_high)
        
        #u low
        ax_u_low.pcolormesh(adcp_data_low['t'], depth_adcp_low, currx_adcp_low.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
        
        #v low
        v_col_low = ax_v_low.pcolormesh(adcp_data_low['t'], depth_adcp_low, curry_adcp_low.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
        
        #u high
        u_col_high = ax_u_high.pcolormesh(adcp_data_high['t'], depth_adcp_high, currx_adcp_high.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
        
        #v high
        v_col_high = ax_v_high.pcolormesh(adcp_data_high['t'], depth_adcp_high, curry_adcp_high.T, cmap=cmocean.cm.balance, vmin=-0.25,vmax=0.25, rasterized = True)
        
        #S2 low
        S2_col_low = ax_S_low.pcolormesh(adcp_data_low['t'], depth_adcp_low, shear2_adcp_low.T, cmap=cmocean.cm.speed, vmin=0,vmax=0.005, rasterized = True)
        
        #S2 high
        S2_col_high = ax_S_high.pcolormesh(adcp_data_high['t'], depth_adcp_high, shear2_adcp_high.T, cmap=cmocean.cm.speed, vmin=0,vmax=0.005, rasterized = True)
    
    # =============================================================================
    # #save
    # =============================================================================
    # fig.savefig(path_results_temp + cruise + mooring + '_wind_uv_shear_spectra_avgd.png', dpi = 300, bbox_inches='tight')
    fig.savefig(path_results_temp + cruise + mooring + '_wind_uv_shear_spectra_avgd.pdf', bbox_inches='tight')

    # pl.show()
    fig.clf()

starttime = timing.time()

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
