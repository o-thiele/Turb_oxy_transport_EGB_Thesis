import numpy as np
import pylab as pl
import matplotlib
import time as timing
import scipy.io as ssio
import gsw
import pickle
import os
#own functions
import own_functions as own

# =============================================================================
MINI_SIZE = 15
SMALL_SIZE = 17
MEDIUM_SIZE = 20
BIGGER_SIZE = 22
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
    year, year_number = own.get_year_metadata(first, second, third)
    lon, lat = own.get_mooring_locations(first, second, third, deep)
    
    print('currently processing: ', first, second, third, deep)
    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    # operating_system = 'windoof'
    if first == True:

        transect_list = ['TS17.mat', 'TS18.mat', 'TS19.mat', 'TS110.mat', 'TS111.mat', 'TS112.mat', 'TS113.mat', 'TS114.mat', 'TS115.mat', 'TS116.mat']
        # transect_list =  ['TS18.mat']
        halocline_center = md.z17.halocline_center
        halocline_interval = md.z17.halocline_interval

    elif second == True:

        transect_list = ['TS1_2.mat', 'TS1_3.mat', 'TS1_4.mat', 'TS1_5.mat', 'TS1_6.mat', 'TS1_7.mat', 'TS1_8.mat', 'TS1_9.mat',
                         'TS1_10.mat', 'TS1_11.mat', 'TS1_12.mat', 'TS1_13.mat', 'TS1_14.mat', 'TS1_15.mat']#, 'TS1_16.mat']
        halocline_center = md.z18.halocline_center
        halocline_interval = md.z18.halocline_interval

    elif third == True:
        transect_list = ['TR1-1.mat', 'TR1-2.mat', 'TR1-3.mat', 'TR1-4.mat', 'TR1-5.mat', 'TR1-6.mat', 'TR1-7.mat', 'TR1-8.mat', 'TR1-9.mat',
                         'TR1-10.mat', 'TR1-11.mat', 'TR1-12.mat', 'TR1-13.mat', 'TR1-14.mat'#, 'S106-1.mat', 'S106-2.mat',
                         #'S106-3.mat', 'S106-4.mat', 'S106-5.mat', 'S106-6.mat', 'S106-7.mat', 'S106-8.mat', 'S106-9.mat'
                         ]
            # transect_list = ['TR1-1.mat', 'TR1-2.mat']
        halocline_center = md.z19.halocline_center
        halocline_interval = md.z19.halocline_interval
    
    # =============================================================================
    #     #load wind data
    # =============================================================================
# =============================================================================
#     winddata_u = np.genfromtxt(path_to_wind + str(year_number) + '_u.dump', comments = '%', skip_header=4)
#     winddata_v = np.genfromtxt(path_to_wind + str(year_number) + '_v.dump', comments = '%', skip_header=4)
# =============================================================================

    # =============================================================================
    # load ADCP data
    # =============================================================================

# =============================================================================
#     adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated.npz', allow_pickle = True)
#     try:
#         adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_interpolated_down.npz', allow_pickle = True)
#     except:
#         pass
# =============================================================================
    
    # =============================================================================
    # MSS Data
    # =============================================================================
    rows = 2
    cols = 1
    fig, (ax,cax) = pl.subplots(rows ,cols, constrained_layout=True, figsize=(7,6), gridspec_kw={'height_ratios': [1,0.05]})
    # pl.clf()

    cmap = pl.cm.twilight
    cutoff_lons = 0.015
    norm = matplotlib.colors.Normalize(vmin=-own.haversine_distance(lat, lon, lat, lon + cutoff_lons), vmax=own.haversine_distance(lat, lon, lat, lon + cutoff_lons))
    colors = cmap(np.linspace(0,1,100))

    for filename in transect_list:
        #initialize arrays to save data:
        #load matlab data:
        matlab = ssio.loadmat(path_to_mss_files + filename)
        CTD_sub = matlab['CTD']
      
        NProf = len(CTD_sub['P'][0])
          
        for i in range(NProf): # loop over profiles
              # depth
              lon_mss = matlab['STA'][0, i][4]
                  
              # if lon < 20.55 or lon > 20.65:#
              if lon_mss[0][0] > lon-cutoff_lons and lon_mss[0][0] < lon + cutoff_lons:#
        
                  d_lon = (lon_mss[0][0]-lon)/(2*cutoff_lons) + 0.5    #distance from mooring from 0 to 1, centered at 0.5; for colorpicking
                  sig0 = CTD_sub['SIGTH'][0, i]
                  z_CTD = gsw.z_from_p(CTD_sub['P'][0,i], lat = lat)
                  ax.plot(sig0,z_CTD, color = colors[int(np.floor(d_lon*100))], alpha = 1-2*(abs(d_lon-0.5)), lw = 2.5)

        
    # =============================================================================
    #     #plot
    # =============================================================================
    # ax.set_title(cruise + '_' + mooring)

    ax.set_ylabel('depth [m]')
    ax.set_xlabel(r'$\sigma_0$ [kg m$^{-3}$]')
    
    # ax.vlines(halocline_center-halocline_interval, ymin = -md.z19.depth_deep, ymax = np.nanmax(z_CTD), ls = '--', label = 'halocline definition')
    # ax.vlines(halocline_center+halocline_interval, ymin = -md.z19.depth_deep, ymax = np.nanmax(z_CTD), ls = '--')
    
    ax.vlines(halocline_center-halocline_interval, ymin = -110, ymax = 0, ls = '--', lw = 2.5, label = 'halocline definition')
    ax.vlines(halocline_center+halocline_interval, ymin = -110, ymax = 0, ls = '--', lw = 2.5)

    
    ax.set_ylim(-110,0)
    
    ax.grid()

    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,orientation='horizontal', label= 'distance from {} mooring in longitude direction [m]'.format(mooring), fraction=0.07)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,orientation='horizontal', label= 'distance from mooring in longitude direction [m]', fraction=0.07)

    ax.legend()

    path_results_temp = path_results + 'plots/mss/'
    if not os.path.exists(path_results_temp) == True:
        os.makedirs(path_results_temp)
    
    pl.savefig(path_results_temp + 'sig0_around_' + mooring  + '_mooring_' + cruise + '.pdf')



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
    