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


def main(first, second, third, deep):
    
    print('currently processing: ', first, second, third, deep)
    
    offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh = own.getADCP_plot_definitions(first, second, third, deep)
    largeTicks = mdates.DayLocator()
    smallTicks = mdates.HourLocator(byhour=[6, 12, 18])
    dayformat = mdates.DateFormatter('%d.%b')

    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    year, year_number = own.get_year_metadata(first, second, third)
    lon_mooring, lat_mooring = own.get_mooring_locations(first, second, third, deep)
    # operating_system = 'windoof'
    
    path_results_temp = path_results + 'adcp/filter/'
    path_results_data = path_ADCP_data + 'npz/'

    
    # =============================================================================
    # for ADCP data
    # =============================================================================

    adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated.npz', allow_pickle = True)
    downward_adcp_data = False
    try:
        adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_interpolated_down_avgd.npz', allow_pickle = True)
        downward_adcp_data = True
    except:
        print('no downward adcp data found')
        pass
    
    # =============================================================================
    t_matlab_adcp = adcp_data['t_matlab']
    t_adcp = adcp_data['t']
    # print(t_adcp[0:6])
    depth_adcp = adcp_data['depth'] 
    mdepth_adcp = adcp_data['mdepth']
    p_adcp = adcp_data['p']
    currx_adcp = adcp_data['currx']
    curry_adcp = adcp_data['curry']
    # =============================================================================
    fs_up = meta[year]['adcp_fs_up']
    
    if downward_adcp_data:
        t_matlab_adcp_down = adcp_data_down['t_matlab']
        t_adcp_down = adcp_data_down['t']
        # print(t_adcp[0:6])
        depth_adcp_down = adcp_data_down['depth'] 
        mdepth_adcp_down = adcp_data_down['mdepth']
        p_adcp_down = adcp_data_down['p']
        currx_adcp_down = adcp_data_down['currx']
        curry_adcp_down = adcp_data_down['curry']
        # =============================================================================
        fs_down = meta[year]['adcp_fs_down']
    
    #cutoff freq:
    hours_filter = 24
    # days_filter = 14
    # fc_lowlow = 1/(60*60*24*days_filter)
    fc_low = 1/(60*60*hours_filter)   
    fc_high = 1/(60*60*hours_filter) 
    
    # =============================================================================
    # #filter
    # =============================================================================
    #high and lowpass:
    u_high = own.butterThis(currx_adcp, fs_up, fc_high, 'high', 2)
    u_low = own.butterThis(currx_adcp, fs_up, fc_low, 'low', 2)
    
    #high and lowpass:
    v_high = own.butterThis(curry_adcp, fs_up, fc_high, 'high', 2)
    v_low = own.butterThis(curry_adcp, fs_up, fc_low, 'low', 2)
    
    if downward_adcp_data:
        #high and lowpass:
        u_high_down = own.butterThis(currx_adcp_down, fs_down, fc_high, 'high', 2)
        u_low_down = own.butterThis(currx_adcp_down, fs_down, fc_low, 'low', 2)
        
        #high and lowpass:
        v_high_down = own.butterThis(curry_adcp_down, fs_down, fc_high, 'high', 2)
        v_low_down = own.butterThis(curry_adcp_down, fs_down, fc_low, 'low', 2)

# =============================================================================
# #save data
# =============================================================================

    np.savez_compressed(path_results_data + 'adcp_low_freqs', t_matlab = t_matlab_adcp, t = t_adcp,
                        depth = depth_adcp, mdepth = mdepth_adcp, p = p_adcp,
                        currx = u_low.T, curry = v_low.T)
    
    np.savez_compressed(path_results_data + 'adcp_high_freqs', t_matlab = t_matlab_adcp, t = t_adcp,
                        depth = depth_adcp, mdepth = mdepth_adcp, p = p_adcp,
                        currx = u_high.T, curry = v_high.T)
    
    if downward_adcp_data:
        np.savez_compressed(path_results_data + 'adcp_low_freqs_down_avgd', t_matlab = t_matlab_adcp_down, t = t_adcp_down,
                            depth = depth_adcp_down, mdepth = mdepth_adcp_down, p = p_adcp_down,
                            currx = u_low_down.T, curry = v_low_down.T)
        
        np.savez_compressed(path_results_data + 'adcp_high_freqs_down_avgd', t_matlab = t_matlab_adcp_down, t = t_adcp_down,
                            depth = depth_adcp_down, mdepth = mdepth_adcp_down, p = p_adcp_down,
                            currx = u_high_down.T, curry = v_high_down.T)

# =============================================================================
#     #plots
# =============================================================================

# =============================================================================
#     rows = 3
#     cols = 1
#     kw = {'height_ratios':[1,1,1]}
#     fig, (ax,ax2,ax3) = pl.subplots(rows,cols,  gridspec_kw=kw, figsize = (12,10), sharex = True)
#     
#     pl_adcp = ax.pcolormesh(adcp_data['t'], depth_adcp, curry_adcp.T, cmap=cmocean.cm.balance, shading='nearest', vmin=-0.25,vmax=0.25)
#     if downward_adcp_data:
#         ax.pcolormesh(adcp_data_down['t'], depth_adcp_down, curry_adcp_down.T, cmap=cmocean.cm.balance, shading='nearest', vmin=-0.25,vmax=0.25)
#     ax.set_ylim([ylow, yhigh])
#     ax.set_ylabel('depth [m]')
#     ax.set_title('original')
#     col = pl.colorbar(pl_adcp, ax = (ax,ax2,ax3), shrink = 0.3, anchor = (0,1), aspect = 12)
#     col.set_label(r'$v [ms^{-1}]$')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
# 
#     pl_adcp = ax2.pcolormesh(adcp_data['t'], depth_adcp, v_high, cmap=cmocean.cm.balance, shading='nearest', vmin=-0.25,vmax=0.25)
#     if downward_adcp_data:
#         ax2.pcolormesh(adcp_data_down['t'], depth_adcp_down, v_high_down, cmap=cmocean.cm.balance, shading='nearest', vmin=-0.25,vmax=0.25)
# 
#     ax2.set_ylim([ylow, yhigh])
#     ax2.set_ylabel('depth [m]')
#     ax2.set_title('high frequencies with periods < {}h'.format(hours_filter))
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax2.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
# 
#     pl_adcp = ax3.pcolormesh(adcp_data['t'], depth_adcp, v_low, cmap=cmocean.cm.balance, shading='nearest', vmin=-0.25,vmax=0.25)
#     if downward_adcp_data:
#         ax3.pcolormesh(adcp_data_down['t'], depth_adcp_down, v_low_down, cmap=cmocean.cm.balance, shading='nearest', vmin=-0.25,vmax=0.25)
#     ax3.set_ylim([ylow, yhigh])
#     ax3.set_ylabel('depth [m]')
#     ax3.set_title('low frequencies with periods > {}h'.format(hours_filter))
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax3.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
#     
#     fig.savefig(path_results_temp + cruise + '_' + mooring + '_filtered_Tc-{}h.png'.format(hours_filter), dpi = 300)
#     
#     pl.show()
#     fig.clf()
# =============================================================================
    
starttime = timing.time()

# cruise_years = ['2019']
cruise_years = ['2017', '2018', '2019']
for cruise_year in cruise_years:
    if cruise_year == '2017':
        main(True, False, False, True)
        main(True, False, False, False)
    elif cruise_year == '2018':
        main(False, True, False, True)
        main(False, True, False, False)
    elif cruise_year == '2019':
        main(False, False, True, True)
        main(False, False, True, False)

endtime = timing.time()
print(endtime - starttime)
