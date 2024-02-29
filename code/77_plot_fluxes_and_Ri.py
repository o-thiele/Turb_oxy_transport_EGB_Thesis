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
# import warnings
# warnings.filterwarnings("ignore")
#own functions
import own_functions as own

#plot params
MINI_SIZE = 9
SMALL_SIZE = 9
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

gridspec_all = {'height_ratios': [1,1,1]}
size_fig = (10,8)
fig, (ax_adcp, ax_flux, ax_Ri) = pl.subplots(3,1, sharex=True, gridspec_kw=gridspec_all, figsize = size_fig)

colors_ax_Ri = ['tab:green', 'tab:blue', 'tab:orange']

def main(first, second, third, deep, filtered = 0):
    print('currently processing: ', first, second, third, deep, filtered)
    
    offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh = own.getADCP_plot_definitions(first, second, third, deep)
    largeTicks = mdates.DayLocator()
    if third == True:
        largeTicks = mdates.DayLocator(interval=2)
    smallTicks = mdates.HourLocator(byhour=[6, 12, 18])
    dayformat = mdates.DateFormatter('%d. %b %Y')
    
    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    year, year_number = own.get_year_metadata(first, second, third)
    lon_mooring, lat_mooring = own.get_mooring_locations(first, second, third, deep)
    # operating_system = 'windoof'
    
    # =============================================================================
    # for ADCP data
    # =============================================================================
    adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated.npz', allow_pickle = True)
    
    downward_adcp_data = False
    try:
        adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_interpolated_down_avgd.npz', allow_pickle = True)
        downward_adcp_data = True
    except:
        pass
        

    # =============================================================================
    t_adcp = adcp_data['t']
    depth_adcp = adcp_data['depth'] 
    currx_adcp = adcp_data['currx']
    curry_adcp = adcp_data['curry']
    shear2_adcp = own.calcShear(currx_adcp, curry_adcp, depth_adcp)
    depth_adcp = depth_adcp + offset_adcp
    if downward_adcp_data:
        t_adcp_down = adcp_data_down['t']
        depth_adcp_down = adcp_data_down['depth'] 
        currx_adcp_down = adcp_data_down['currx']
        curry_adcp_down = adcp_data_down['curry']
        shear2_adcp_down = own.calcShear(currx_adcp_down, curry_adcp_down, depth_adcp_down)
        depth_adcp_down = depth_adcp_down + offset_adcp_down

    
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
        # if plotPresentation == True:
        #     transect_list = ['TR1-2.mat', 'TR1-9.mat']
        if deep == True:
            transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8_2.mat', 'TR1-9.mat',
                         'TR1-10.mat', 'TR1-11.mat', 'TR1-12.mat', 'TR1-13.mat', 'TR1-14.mat'#, 'S106-1.mat', 'S106-2.mat',
                         #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                         ]
        else: transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8_2.mat', 'TR1-9.mat',
                                 'TR1-10.mat'#, 'TR1-11.mat', 'S106-1.mat', 'S106-2.mat',
                                 #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                                 ]
        # transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8.mat', 'TR1-9.mat', 'TR1-10.mat']
        halocline_center = md.z19.halocline_center
        halocline_interval = md.z19.halocline_interval
        fs_up = md.z19.adcp_fs_up
        fs_down = md.z19.adcp_fs_down

# =============================================================================
#     #load fluxes and Ri
# =============================================================================
    path_load_fluxes = path_results + 'fluxes_and_Ri/' + cruise + '_fluxes_Ri/'

    flux_and_Ri_data = np.load(path_load_fluxes + cruise + mooring + 'fluxes_Ri_cruise_filtered_{}.npz'.format(filtered), allow_pickle = True) 
    fluxes_cruise = flux_and_Ri_data['fluxes_cruise']
    fluxes_cruise_std = flux_and_Ri_data['fluxes_cruise_std']
    Ri_cruise = flux_and_Ri_data['Ri_cruise']
    Ri_cruise_std = flux_and_Ri_data['Ri_cruise_std']
    time_start = flux_and_Ri_data['time_start_closest']
    
    # print(fluxes_cruise)
    # print(fluxes_cruise_std)
    # print(fluxes_cruise)
    
    # =============================================================================
    # #plot
    # =============================================================================
    if filtered == 0:
        #S2
        S2_col = ax_adcp.pcolormesh(adcp_data['t'], depth_adcp, shear2_adcp.T, cmap=cmocean.cm.speed, vmin=0,vmax=0.005, rasterized = True)
        if downward_adcp_data:
            ax_adcp.pcolormesh(adcp_data_down['t'], depth_adcp_down, shear2_adcp_down.T, cmap=cmocean.cm.speed, vmin=0,vmax=0.005, rasterized = True)
        ax_adcp.set_ylim([ylow, yhigh])
        ax_adcp.set_ylabel('depth [m]')
        ax_adcp.scatter(time_start, -1*np.ones(len(time_start)),marker = 'v', color = 'black')
        for count, transect_name in enumerate(transect_list):
            ax_adcp.text(time_start[count], -1, transect_name[:-4], 
             color = 'black', rotation = 50, 
             rotation_mode = 'anchor')

        #todo: add schräge titel aus der transect list über die dreiecke!
        
        
        # ax.set_title('high frequencies with periods < {}h'.format(hours_filter))
        own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
        ax_adcp.set_xlim([mdates.datetime.datetime(year_number, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year_number, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
        S2_colbar = pl.colorbar(S2_col, pad = 0.01,  ax = (ax_adcp, ax_flux, ax_Ri), shrink = 0.295, anchor = (0,1), aspect = 12)
        S2_colbar.set_label(r'$S^2[s^{-2}]$')

    # ax_flux.set_ylabel(r'$F_{O2} [mmol \ m^{-2}d^{-1}]$')
        ax_flux.scatter(time_start, fluxes_cruise, color = colors_ax_Ri[0])
        ax_flux.errorbar(time_start, fluxes_cruise, yerr = fluxes_cruise_std, color = colors_ax_Ri[0], fmt = 'o')
    ax_flux.set_ylabel(r'$\langle F_{O2} \rangle [mmol \ m^{-2}d^{-1}]$') 
    ax_flux.set_ylim(top=0)
    ax_flux.grid(visible = True)
    
    # ax_Ri.scatter(time_start, Ri_cruise, color = colors_ax_Ri[filtered])
    if filtered == 2:
        ax_Ri.errorbar(time_start, Ri_cruise, yerr = Ri_cruise_std, fmt = 'o', color = colors_ax_Ri[filtered])  #elinewidth = 0.5
    else:
        ax_Ri.errorbar(time_start, Ri_cruise, yerr = Ri_cruise_std, fmt = 'o',  color = colors_ax_Ri[filtered]) #elinewidth = 1.5,
    ax_Ri.set_ylabel(r'$Ri []$')
    ax_Ri.grid(visible = True)
    legend_elements_ax_Ri = [Line2D([0], [0], marker='o', color = colors_ax_Ri[0], label = r'no filter'), 
                    Line2D([0], [0], marker='o', color = colors_ax_Ri[1], label = r'low-pass'),
                    Line2D([0], [0], marker='o', color = colors_ax_Ri[2], label = r'high-pass')]#,
                    #Line2D([0], [0], ls = '--', color = 'tab:grey', label = '0.25')]
    ax_Ri.legend(handles=legend_elements_ax_Ri, loc='lower left')
    ax_Ri.set_yscale('log')
    # ax_Ri.set_ylim(bottom=0.1)
    xlims = ax_Ri.get_xlim()
    # if filtered == 2:
    #     ax_Ri.hlines(0.25, xmin = xlims[0], xmax = xlims[1], ls = '--', color = 'tab:grey')
    # ax_Ri.legend()
    
    ax_names = (ax_adcp, ax_flux, ax_Ri)
    for count, ax_name in enumerate(ax_names):
        own.plotLetter(ax_name, 1, 0.85, count)
    
    path_results_temp = path_results + 'plots/fluxes_Ri/' 
    if not os.path.exists(path_results_temp) == True:
        os.makedirs(path_results_temp)
    
    
    # fig.savefig(path_results + 'plots/fluxes_Ri/' + cruise + mooring + '_shear_flux_Ri_avgd.png', dpi = 300, bbox_inches='tight')
    fig.savefig(path_results + 'plots/fluxes_Ri/' + cruise + mooring + '_shear_flux_Ri_avgd.pdf', bbox_inches='tight')

    # pl.clf()

starttime = timing.time()

filtered_list = [0, 1, 2]

# =============================================================================
# #this has to be done for each mooring seperately!
# =============================================================================
cruise_years = ['2019']

for cruise_year in cruise_years:
    if cruise_year == '2017':
        for filtered in filtered_list:
            main(True, False, False, True, filtered)
    elif cruise_year == '2018':
        for filtered in filtered_list:
            main(False, True, False, True, filtered)
    elif cruise_year == '2019':
        for filtered in filtered_list:
            # main(False, False, True, True, filtered)
            main(False, False, True, False, filtered)


endtime = timing.time()
print(endtime - starttime)
