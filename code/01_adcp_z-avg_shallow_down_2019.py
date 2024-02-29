import numpy as np
from scipy.interpolate import interp1d
#own functions
import own_functions as own

first = False
second = False
third = True

deep = False

print('currently processing: ', first, second, third, deep)
path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
path_results_temp = path_ADCP_data + 'npz/'
# operating_system = 'windoof'
if first == True:
    year = 2017

elif second == True:
    year = 2018

elif third == True:
    year = 2019

# =============================================================================
# for ADCP data
# =============================================================================

# adcp_data = np.load(path_ADCP_data + 'npz/adcp_interpolated.npz', allow_pickle = True)
try:
    adcp_data_down = np.load(path_ADCP_data + 'npz/adcp_interpolated_down.npz', allow_pickle = True)
    t_matlab_down = adcp_data_down['t_matlab']
    t_adcp_down = adcp_data_down['t']
    depth_adcp_down = adcp_data_down['depth'] 
    mdepth_adcp_down = adcp_data_down['mdepth']
    p_adcp_down = adcp_data_down['p']
    currx_adcp_down = adcp_data_down['currx']
    curry_adcp_down = adcp_data_down['curry']
except:
    print('no downward adcp data found')
    pass
        

#test for reducing and averaging shallow mooring downward looking adcp to 1m bins as upward looking adcp:
# =============================================================================
# depth_crop = adcp_data_down['depth'][2:]
# depth_red = np.mean(depth_crop.reshape(-1, 4), axis=1)
# 
# curry_crop = adcp_data_down['curry'][:,2:]
# curry_red = np.zeros((len(curry_crop[:,0]), len(depth_red)))
# for count in range(len(curry_red[:,0])):
#     curry_red[count,:] = np.mean(curry_crop[count,:].reshape(-1, 4), axis=1)
#     
# currx_crop = adcp_data_down['currx'][:,2:]
# currx_red = np.zeros((len(currx_crop[:,0]), len(depth_red)))
# for count in range(len(currx_red[:,0])):
#     currx_red[count,:] = np.mean(currx_crop[count,:].reshape(-1, 4), axis=1)
# =============================================================================

#approach with interp1d
new_depths = np.arange(-68., -78., -1.)

NProf = len(currx_adcp_down[:,0])
    # number of adcp entries

# Initialize the array to store interpolated density values
currx_red = np.zeros((len(new_depths), NProf)) # Z is the depth, NProf is the number of profiles
curry_red = np.zeros((len(new_depths), NProf)) # Z is the depth, NProf is the number of profiles

for i in range(NProf): # loop over profiles
        # depth
        # Perform the interpolation
    interp_func_x = interp1d(depth_adcp_down, currx_adcp_down[i,:], bounds_error=False, fill_value=np.nan)
    currx_red[:, i] = interp_func_x(new_depths) 
    interp_func_y = interp1d(depth_adcp_down, curry_adcp_down[i,:], bounds_error=False, fill_value=np.nan)
    curry_red[:, i] = interp_func_y(new_depths) 
    

print('saving averaged data..')
# np.savez_compressed(path_results_temp + 'adcp_interpolated', t_matlab = t_matlab, t = timex_cols,
#                     depth = depthx_rows, mdepth = mdepth, p = p,
#                     currx = currx_interp, curry = curry_interp)
try:
    np.savez_compressed(path_results_temp + 'adcp_interpolated_down_avgd', t_matlab = t_matlab_down, t = t_adcp_down,
                        depth = new_depths, mdepth = mdepth_adcp_down, p = p_adcp_down,
                        currx = currx_red.T, curry = curry_red.T)
except:
    pass