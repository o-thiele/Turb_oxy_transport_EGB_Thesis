import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import pylab as pl
import os
import matplotlib.dates as mdates
import time as timing
import pickle
#own functions
import own_functions as own

#plot params
MINI_SIZE = 13
SMALL_SIZE = 15
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
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

    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    year, year_number = own.get_year_metadata(first, second, third)
    lon_mooring, lat_mooring = own.get_mooring_locations(first, second, third, deep)
    # operating_system = 'windoof'
    
    path_results_temp = path_results + 'plots/adcp/spectra/' + year + '/'
    if not os.path.exists(path_results_temp) == True:
        os.makedirs(path_results_temp)

    
    # =============================================================================
    # for ADCP data
    # =============================================================================
    
    downward_adcp_data = False
    adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated.npz', allow_pickle = True)
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
#     length = len(t_adcp) + 6000
#     cutoff = -1#15
#     print(length)
#     
#     t_adcp = t_adcp[round(length/2):]
#     currx_adcp = currx_adcp[round(length/2):,:cutoff]
#     curry_adcp = curry_adcp[round(length/2):,:cutoff]
#     depth_adcp = depth_adcp[:cutoff]
#     print(depth_adcp)
# 
#     print(len(t_adcp))
# =============================================================================
    
    depth_adcp = depth_adcp + offset_adcp
    # =============================================================================
    if downward_adcp_data:
        t_adcp_down = adcp_data_down['t']
        # print(t_adcp[0:6])
        depth_adcp_down = adcp_data_down['depth'] 
        mdepth_adcp_down = adcp_data_down['mdepth']
        p_adcp_down = adcp_data_down['p']
        currx_adcp_down = adcp_data_down['currx']
        curry_adcp_down = adcp_data_down['curry']
        
        depth_adcp_down = depth_adcp_down + offset_adcp_down
        
    fs_up = meta[year]['adcp_fs_up']
    fs_down = meta[year]['adcp_fs_down']
        # fs_up = fs_down
    
    
    # =============================================================================
    #     #cut off depth index
    # =============================================================================
# =============================================================================
#     print(depth_adcp)
#     depth_index_low =0
#     depth_index_high = depth_index_low + 15
#     
#     # depth_adcp = depth_adcp[depth_index:]
#     # currx_adcp = currx_adcp[:,depth_index:]
#     # curry_adcp = curry_adcp[:,depth_index:]
#     
#     # depth_adcp = depth_adcp[0:depth_index]
#     # currx_adcp = currx_adcp[:,0:depth_index]
#     # curry_adcp = curry_adcp[:,0:depth_index]
#     
#     depth_adcp = depth_adcp[depth_index_low:depth_index_high]
#     currx_adcp = currx_adcp[:,depth_index_low:depth_index_high]
#     curry_adcp = curry_adcp[:,depth_index_low:depth_index_high]
#     
#     print(depth_adcp)
# =============================================================================
    
    # =============================================================================
    # #spectrum by welch
    # =============================================================================

    npersegment = len(currx_adcp[:,0]) - round(len(currx_adcp[:,0])/4)
    # print(npersegment)
    f, P_welch_u = own.calcSpectrumWelch(currx_adcp, fs_up, npersegment)
    f, P_welch_v = own.calcSpectrumWelch(curry_adcp, fs_up, npersegment)
    f[:] = f[:]*3600 #cph
    
    
    f_col_x, P_welch_u_color = own.calcSpectrumWelchColor(currx_adcp, fs_up, npersegment)
    f_col_y, P_welch_v_color = own.calcSpectrumWelchColor(curry_adcp, fs_up, npersegment)


    #rotary spectrum:
    U = currx_adcp + curry_adcp * 1j
    freqsRotary, ps_U = own.calcSpectrumRotary(U, fs_up, npersegment)
    # freqsRotary = np.linspace(-1/(2*time_step), 1/(2*time_step), len(ps_U))
    freqsRotary[:] = freqsRotary[:]*3600    #cycles per hour, cph
    # print(freqsRotary)
    for count, item in enumerate(freqsRotary):
        if item < 0:
            # print(count)
            ps_clock = ps_U[count:]
            ps_anticlock = ps_U[1:count]
            freqs_clock = -1 * freqsRotary[count:]
            freqs_anticlock = freqsRotary[1:count]
            break

    
    omega = 1/86400
    f_inertial = 2 * omega * np.sin(np.pi/180 *meta[year]['lat_deep'])
# =============================================================================
#     print(meta[year]['lat_deep'])
#     print(np.sin(meta[year]['lat_deep']))
#     print(np.sin(np.pi/180 *meta[year]['lat_deep']))
#     print('f_inertial :', f_inertial)
#     print(1/f_inertial)
#     
#     print(1/f_inertial /60 /60)
# =============================================================================
    
    #downwards
    if downward_adcp_data:
        npersegment_down = len(currx_adcp_down[:,0]) - round(len(currx_adcp[:,0])/4)#/2
        # print(npersegment)
        f_down, P_welch_u_down = own.calcSpectrumWelch(currx_adcp_down, fs_down, npersegment_down)
        f_down, P_welch_v_down = own.calcSpectrumWelch(curry_adcp_down, fs_down, npersegment_down)
        f_down[:] = f_down[:]*3600 #cph
        
        
        f_col_x_down, P_welch_u_color_down = own.calcSpectrumWelchColor(currx_adcp_down, fs_down, npersegment_down)
        f_col_y_down, P_welch_v_color_down = own.calcSpectrumWelchColor(curry_adcp_down, fs_down, npersegment_down)
    
    
        #rotary spectrum:
        U_down = currx_adcp_down + curry_adcp_down * 1j
        freqsRotary_down, ps_U_down = own.calcSpectrumRotary(U_down, fs_down, npersegment_down)
        # freqsRotary = np.linspace(-1/(2*time_step), 1/(2*time_step), len(ps_U))
        freqsRotary_down[:] = freqsRotary_down[:]*3600    #cycles per hour, cph
        for count, item in enumerate(freqsRotary_down):
            if item < 0:
                # print(count)
                ps_clock_down = ps_U_down[count:]
                ps_anticlock_down = ps_U_down[1:count]
                freqs_clock_down = -1 * freqsRotary_down[count:]
                freqs_anticlock_down = freqsRotary_down[1:count]
                break

# =============================================================================
#     #plots
# =============================================================================

    # fig = pl.figure(1, figsize =(12,5))
    # pl.clf()
    # fig.suptitle(cruise + ' ' + mooring + ' --- Shear and other plot', fontsize=14)
    rows = 1
    cols = 1
    figsizes = (7,5)
    # kw = {'height_ratios':[1,1,1,1]}
    
# =============================================================================
#     #welch u
# =============================================================================
# =============================================================================
#     fig, ax = pl.subplots(rows,cols, figsize = (7,5))
#     
#     # ax.plot(freqs, ps_y, label = 'np - v')
#     # ax.plot(f, P_welch_v, label = 'welch - v')
#     
#     ax.plot(freqs, ps_x, label = 'np - u')
#     ax.plot(f, P_welch_u, label = 'welch -u')
#     # ax2.vlines(1/meta[year]['adcp_cutoff_low'], ymin = 0, ymax = max(ps_y), ls = '--', label = 'cutoff - low = {}h'.format(meta[year]['adcp_cutoff_low']))  #cph counts per hour
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.vlines(f_inertial*3600, ymin = 0, ymax = max(ps_y), label = 'inertial freq', color = 'tab:green')  #cph counts per hour
#     ax.vlines(1/3/24, ymin = 0, ymax = max(ps_y), ls = '--', label = '3d', color = 'tab:green')  #cph counts per hour
# 
#     ax.legend()
#     
#     fig.savefig(path_results_temp + mooring + '_spectra_welch_u.png', dpi = 300)
#     
#     pl.show()
#     fig.clf()
# =============================================================================

# =============================================================================
#     #welch v
# =============================================================================
# =============================================================================
#     fig, ax = pl.subplots(rows,cols, figsize = (7,5))
#     
#     ax.plot(freqs, ps_y, label = 'np - v')
#     ax.plot(f, P_welch_v, label = 'welch - v')
#     
#     # ax.plot(freqs, ps_x, label = 'np - u')
#     # ax.plot(f, P_welch_u, label = 'welch -u')
#     # ax2.vlines(1/meta[year]['adcp_cutoff_low'], ymin = 0, ymax = max(ps_y), ls = '--', label = 'cutoff - low = {}h'.format(meta[year]['adcp_cutoff_low']))  #cph counts per hour
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.vlines(f_inertial*3600, ymin = 0, ymax = max(ps_y), label = 'inertial freq', color = 'tab:green')  #cph counts per hour
#     ax.vlines(1/3/24, ymin = 0, ymax = max(ps_y), ls = '--', label = '3d', color = 'tab:green')  #cph counts per hour
# 
#     ax.legend()
#     
#     fig.savefig(path_results_temp + mooring + '_spectra_welch_v.png', dpi = 300)
#     
#     pl.show()
#     fig.clf()
# =============================================================================

# =============================================================================
#     #welch combined
# =============================================================================

# =============================================================================
#     fig, ax = pl.subplots(rows,cols, figsize = figsizes)
#     
#     ax.plot(freqs, ps_x + ps_y, label = 'np')
#     ax.plot(f, P_welch_v + P_welch_u , label = 'welch')
#     
#     if downward_adcp_data:
#         ax.plot(freqs_down, ps_x_down + ps_y_down, label = 'np - down')
#         ax.plot(f_down, P_welch_v_down + P_welch_u_down , label = 'welch - down')
#     # ax.plot(freqs, ps_x, label = 'np - u')
#     # ax.plot(f, P_welch_u, label = 'welch -u')
#     # ax2.vlines(1/meta[year]['adcp_cutoff_low'], ymin = 0, ymax = max(ps_y), ls = '--', label = 'cutoff - low = {}h'.format(meta[year]['adcp_cutoff_low']))  #cph counts per hour
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.vlines(f_inertial*3600, ymin = 0, ymax = max(ps_y), label = 'inertial freq', color = 'tab:green')  #cph counts per hour
#     ax.vlines(1/3/24, ymin = 0, ymax = max(ps_y), ls = '--', label = '3d', color = 'tab:green')  #cph counts per hour
# 
#     ax.legend()
#     
#     fig.savefig(path_results_temp + mooring + '_spectra_welch_combined_avgd.png', dpi = 300, bbox_inches='tight')
#     
#     pl.show()
#     fig.clf()
# =============================================================================
    
# =============================================================================
#     #rotary
# =============================================================================
    fig, ax = pl.subplots(rows,cols, figsize = figsizes)
    
    ax.plot(freqs_clock, ps_clock, label = 'clockwise')
    ax.plot(freqs_anticlock, ps_anticlock, label = 'anticlockwise')
    
    # ax.plot(freqs_clock_like, ps_clock_like, label = 'like clockwise')
    # ax.plot(freqs_anticlock_like, ps_anticlock_like, label = 'like anticlockwise')
    # print(ps_clock_like)
    if downward_adcp_data:
        ax.plot(freqs_clock_down, ps_clock_down, color = 'tab:green', label = 'clockwise - down')
        ax.plot(freqs_anticlock_down, ps_anticlock_down, color = 'tab:red', label = 'anticlockwise - down')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    linewidth = 1.5
    linecolor = 'tab:grey'
    line_lims = ax.get_ylim()
    line_upper_lim = line_lims[1]
    # print(line_upper_lim)
    
    ax.vlines(f_inertial*3600, ymin = 0, ymax = line_upper_lim, label = r'$T_\text{inertial}$', lw = linewidth, color = linecolor)  #cph counts per hour
    ax.vlines(1/24, ymin = 0, ymax = line_upper_lim, ls = '--', label = r'$T = 1$ day', lw = linewidth, color = linecolor)  #cph counts per hour
    ax.vlines(1/3/24, ymin = 0, ymax = line_upper_lim, ls = '-.', label = r'$T = 3$ days', lw = linewidth, color = linecolor)  #cph counts per hour

    ax.set_xlabel(r'$\omega \ [\text{cph}]$')
    ax.set_ylabel(r'PSD $[(m s^{-1})^{2} \text{cph}^{-1}]$')

    ax.legend()
    ax.grid()
    
    def fToT(f):
        #takes f in cycles per hour
        #returns T in hours
        T = 1/f 
        return T
    
    def Ttof(T):
        f = 1/T
        return f
    
    ax_T = ax.secondary_xaxis('top', functions=(fToT, Ttof))
    ax_T.set_xlabel('$T \ [h]$')
    
# =============================================================================
#     ax_T = ax.twiny()
#     ax_T.set_xlim(ax.get_xlim())
#     label_format= '%.3f'
#     inverse_ticks = []
#     for tick in ax.get_xticks():
#         tick_T = 2*np.pi/tick
#         inverse_ticks.append(label_format % (tick_T,))
#     ax_T.set_xticklabels(inverse_ticks)
# =============================================================================
        
    # fig.savefig(path_results_temp + mooring + '_spectra_rotary_avgd.png', dpi = 300, bbox_inches='tight')
    fig.savefig(path_results_temp + mooring + '_spectra_rotary_avgd.pdf', bbox_inches='tight')
    
    pl.show()
    fig.clf()
    

# =============================================================================
#     #color combined
# =============================================================================
    fig, ax4 = pl.subplots(rows,cols, figsize = figsizes)
    # f_col_x, P_welch_u_color 
    
# =============================================================================
#     ps_col = ps_x_col + ps_y_col
#     if downward_adcp_data:
#         ps_col_down = ps_x_col_down + ps_y_col_down
#         
#     col_min = 0.001
#     col_max = 1e5
#     ylim = [depth_adcp[0]-1, 0]
# =============================================================================
        
    ps_col = P_welch_u_color + P_welch_v_color
    if downward_adcp_data:
        ps_col_down = P_welch_u_color_down + P_welch_v_color_down

    col_min = 0.001
    col_max = 2*1e3

    ax4.remove()
    ax4 = pl.subplot(111, sharex=ax)
    # pl_spectrum = ax4.pcolormesh(f_col_y, depth_adcp, P_welch_v_color.T, shading='nearest', norm=matplotlib.colors.LogNorm())
    
    # pl_spectrum = ax4.pcolormesh(freqs, depth_adcp, ps_col.T, shading='nearest', norm=matplotlib.colors.LogNorm(vmin = col_min, vmax = col_max))
    pl_spectrum = ax4.pcolormesh(f_col_x*3600, depth_adcp, ps_col.T, shading='nearest', rasterized = True, norm=matplotlib.colors.LogNorm(vmin = col_min, vmax = col_max))

    if downward_adcp_data:
        # pl_spectrum_down = ax4.pcolormesh(freqs_down, depth_adcp_down, ps_col_down.T, shading='nearest', norm=matplotlib.colors.LogNorm(vmin = col_min, vmax = col_max))
        pl_spectrum_down = ax4.pcolormesh(f_col_x_down*3600, depth_adcp_down, ps_col_down.T, shading='nearest', rasterized = True, norm=matplotlib.colors.LogNorm(vmin = col_min, vmax = col_max))


    col2 = pl.colorbar(pl_spectrum, ax = ax4)
    col2.set_label(r'PSD $[(m s^{-1})^{2} \text{cph}^{-1}]$')
    ax4.set_xlabel(r'$\omega \ [\text{cph}]$')
    ax4.set_ylabel(r'depth $[m]$')
    # col2_down = pl.colorbar(pl_spectrum_down, ax = ax4)
    #set same colorrange
    # col2_down.mappable.set_clim(*col2.mappable.get_clim())
    # col2_down.remove() 
    # ax4.set_xlim([0.001, 0.1])

    ax4.set_xscale('log')

    
    if downward_adcp_data:
        ax4.vlines(f_inertial*3600, ymin = depth_adcp_down[-1]-1, ymax = 0, label = r'$T_\text{inertial}$', lw = linewidth, color = linecolor)  #cph counts per hour
        ax4.vlines(1/24, ymin = depth_adcp_down[-1]-1, ymax = 0, ls = '--', label = r'$T = 1$ day', lw = linewidth, color = linecolor)  #cph counts per hour
        ax4.vlines(1/3/24, ymin = depth_adcp_down[-1]-1, ymax = 0, ls = '-.', label = r'$T = 3$ days', lw = linewidth, color = linecolor)  #cph counts per hour

        ylim = [depth_adcp_down[-1]-1, 0]
    else:
        ax4.vlines(f_inertial*3600, ymin = depth_adcp[0]-1, ymax = 0, label = r'$T_\text{inertial}$', lw = linewidth, color = linecolor)  #cph counts per hour
        ax4.vlines(1/24, ymin = depth_adcp[0]-1, ymax = 0, ls = '--', label = r'$T = 1$ day', lw = linewidth, color = linecolor)  #cph counts per hour
        ax4.vlines(1/3/24, ymin = depth_adcp[0]-1, ymax = 0, ls = '-.', label = r'$T = 3$ days', lw = linewidth, color = linecolor)  #cph counts per hour

        ylim = [depth_adcp[0]-1, 0]
    ax4.set_ylim(ylim)
    ax4.legend()
    
    ax4_T = ax4.secondary_xaxis('top', functions=(fToT, Ttof))
    ax4_T.set_xlabel(r'$T \ [h]$')


    # fig.savefig(path_results_temp + mooring + '_spectra_color_combined_avgd.png', dpi = 300, bbox_inches='tight')
    fig.savefig(path_results_temp + mooring + '_spectra_color_combined_avgd.pdf', bbox_inches='tight')
    
    pl.show()
    fig.clf()

starttime = timing.time()

# cruise_years = ['2019']#, '2019']
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
