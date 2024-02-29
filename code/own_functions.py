import scipy.io as ssio
import datetime as dt
import locale
from datetime import datetime
import numpy as np
import gsw
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
import pickle
import string
from scipy.integrate import cumulative_trapezoid

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


def getPaths(mode = 'relative', first = False, second = False, third = False, deep = True):
    
    if deep == True:
        path_mooring = 'TC_deep/'
        mooring = 'deep'
    else:
        path_mooring = 'TC_shallow/'
        mooring = 'shallow'
        
    if first == True:
        path_cruise = 'EMB169/'
    elif second == True:
        path_cruise = 'EMB177/'
    elif third == True:
        path_cruise = 'EMB217/'
            
    if mode == 'relative':
        operating_system = 'linux'
        #may need to change the operating system bc some start and endtime calculations for mss casts involve machine locale settings
        #other option: 'windoof'
        groundpath = '../'
        groundpath_mss = groundpath + 'data_mss/mss_data_all/'
        path_mss_interpolated = groundpath + 'results/mss/mss_interpolated/'
        
        path_to_wind = groundpath + 'winddata_meteo/'
        finepathADCP = 'data_adcp_and_moorings/' + path_cruise + path_mooring
        
        path_results = groundpath + 'results/'
                
            
    path_ADCP_data = groundpath + finepathADCP + 'adcp/data/'
    path_to_mss_files = groundpath_mss + path_cruise
    cruise = path_cruise[:-1]
            
    return path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system

def get_year_metadata(first, second, third):
    if first:
        year = '2017'
        year_number = 2017
    elif second:
        year = '2018'
        year_number = 2018
    else:
        year = '2019'
        year_number = 2019
    return year, year_number

def get_mooring_locations(first, second, third, deep):
    if first == True:
        if deep:
            lon = md.z17.lon_deep
            lat = md.z17.lat_deep
        else:
            lon = md.z17.lon_shallow
            lat = md.z17.lat_shallow
    elif second == True:
        if deep:
            lon = md.z18.lon_deep
            lat = md.z18.lat_deep
        else:
            lon = md.z18.lon_shallow
            lat = md.z18.lat_shallow
    elif third == True:
        if deep:
            lon = md.z19.lon_deep
            lat = md.z19.lat_deep
        else:
            lon = md.z19.lon_shallow
            lat = md.z19.lat_shallow         
    return lon, lat


def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    radius_earth = 6371.0  # Radius of the Earth in kilometers
    distance_km = radius_earth * c

    # Convert distance from kilometers to meters
    distance_m = distance_km * 1000

    return distance_m

def matlabTimeConversion(pl, major_locator, date_fmt, minor_locator):
    ax = pl.gca()
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_minor_locator(minor_locator)

def secondsUntil(year):
    #calculate number of seconds that went by until the passed year
    days = 0
    for y in range(year):

        if y % 4 == 0:
            if ((y % 100 == 0) and (y % 400 != 0)):
                days += 365
            else:
                days += 366
        else:
            days += 365
        # print(y, days)
    seconds = days * 24*60*60
    return seconds


def plotLetter(ax_name, x,y,plot_number, size = 20, weight='bold', ha='right'):
    #plot letter in ax_name of plot, plot_number 0 = A, 1 = B, etc.
    ax_name.text(x, y, string.ascii_lowercase[plot_number] + ')', transform=ax_name.transAxes, size=size, weight=weight, ha=ha)

def calcStartAndEndtime(filepath, castnumber):
    #calculates the start and endtime of an mss cast from a single cast. 
    #Therefore the locale of the computer might have to be changed
    matlab = ssio.loadmat(filepath)
    mss_time_start_string = str(matlab['STA'][0,castnumber][2][0])      #as string, e.g. '22-Oct-2017 04:49:53'
    # locale.setlocale(locale.LC_ALL, 'en_US') #windows
    locale.setlocale(locale.LC_ALL, 'en_US.utf8') #linux
    mss_time_start = datetime.strptime(mss_time_start_string, '%d-%b-%Y %H:%M:%S')
    # locale.setlocale(locale.LC_ALL, 'de_DE')
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')

    try:
        mss_duration = len(matlab['DATA'].T[castnumber][0][0][0])/(matlab['PAR']['Fs'][0][0][0][0]/matlab['PAR']['lineav'][0][0][0][0])
    except KeyError:    #for files from 2019
        mss_duration = len(matlab['DATA'].T[castnumber][0][0][0])/256
    mss_time_end = mss_time_start + dt.timedelta(0,mss_duration) # days, seconds, then other fields.
    return mss_time_start, mss_time_end

def calcStartAndEndtime_singleCast(mss_file, castnumber, operating_system = 'linux'):
    #calculates the start and endtime of an mss cast from a single cast. 
    #Therefore the locale of the computer might have to be changed
    mss_time_start_string = str(mss_file['STA'][0,castnumber][2][0])      #as string, e.g. '22-Oct-2017 04:49:53'

    if operating_system == 'windoof':
        locale.setlocale(locale.LC_ALL, 'en_US') #windows
    else:
        locale.setlocale(locale.LC_ALL, 'en_US.utf8') #linux
    mss_time_start = datetime.strptime(mss_time_start_string, '%d-%b-%Y %H:%M:%S')
    if operating_system == 'windoof':
        locale.setlocale(locale.LC_ALL, 'de_DE')
    # elif operating_system == 'linux':
    #     locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
    # else:
    #     pass

    try:
        mss_duration = len(mss_file['DATA'].T[castnumber][0][0][0])/(mss_file['PAR']['Fs'][0][0][0][0]/mss_file['PAR']['lineav'][0][0][0][0])
    except KeyError:    #for files from 2019
        mss_duration = len(mss_file['DATA'].T[castnumber][0][0][0])/256
    mss_time_end = mss_time_start + dt.timedelta(0,mss_duration) # days, seconds, then other fields.
    
    return mss_time_start, mss_time_end


def nearest_ind(items, pivot):
    #find nearest index of items around a pivot
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0)

def nearest_ind_nans(items, pivot):
    #find nearest index of items around a pivot with nans
    difference = np.abs([item - pivot for item in items])
    return np.nanargmin(difference)

def z_from_p(pressure_values):    
    #calculates depth from pressure
    center_gotland_basin_lat = 57.0
    return gsw.z_from_p(pressure_values,center_gotland_basin_lat * np.ones(np.shape(pressure_values)))

def calcRichardson(Nsquared_gsw, z_mid_gsw, shear_adcp_squared, depth):
    #calculates Richardson number and averages the Nsquared array according of the adcp bin depths
    Ri_df = pd.DataFrame()
    Ri_df['Nsquared'] = Nsquared_gsw
    Ri_df['z_mid_gsw'] = z_mid_gsw
                                    
    Nsquared_gsw_avgd = np.copy(shear_adcp_squared)
    z_mid_gsw_avgd = np.copy(Nsquared_gsw_avgd)

    for i, val in enumerate(Nsquared_gsw_avgd):
        Ri_df_filtered = Ri_df.loc[(Ri_df.z_mid_gsw < depth[i] + abs(depth[1] - depth[0])/2)]
        Ri_df_filtered = Ri_df_filtered.loc[(Ri_df_filtered.z_mid_gsw > depth[i] - abs(depth[1] - depth[0])/2)]
        Nsquared_gsw_avgd[i] = np.mean(Ri_df_filtered['Nsquared'])
        z_mid_gsw_avgd[i] = np.mean(Ri_df_filtered['z_mid_gsw'])
    
    Ri = Nsquared_gsw_avgd[:]/shear_adcp_squared[:]      
    
    return Ri, Nsquared_gsw_avgd, z_mid_gsw_avgd


def interp_MSS_var(DICT, Z, varinterp): 
    #interpolates a variable from MSS data
    #DICT is the dictionary of the variable, Z is the depth, varinterp is the variable to be interpolated
    
    NProf = len(DICT['P'][0])
    # number of profiles

    # Initialize the array to store interpolated density values
    Var_interp = np.zeros((len(Z), NProf)) # Z is the depth, NProf is the number of profiles

    for i in range(NProf): # loop over profiles
        # depth
        ZTmp = -DICT['P'][0][i] 
        Ind = ~np.isnan(ZTmp) # find the index of non-nan values
        ZTmp = ZTmp[Ind] # remove nan values
        
        Var = DICT[varinterp][0][i][Ind] #varinterp is the variable to be interpolated, Ind is the index of non-nan values
        # Perform the interpolation
        interp_func = interp1d(ZTmp, Var, bounds_error=False, fill_value=np.nan)
        Var_interp[:, i] = interp_func(Z) 

    return Var_interp

# =============================================================================
# #O2 and flux handling
# =============================================================================
def load_O2_shifted(path, file, count):
    #load shifted O2 data from MSS paper
    # print(path, file)
    lon = 20.5
    lat = 57.3
    matlab = ssio.loadmat(path + 'oxy_shift/' + file[:-4] + '_TODL_merged_shift_oxy.mat')
    TODL_sub = matlab['TODL_MSS_oxy']
    O2_all = TODL_sub['oxy'][0][count]
    O2_mumol = O2_all[:,4] #mu mol / l      #TODO: or /kg?
    P_TODL = TODL_sub['P'][0][count]
    T_TODL = TODL_sub['T'][0][count]
    SP_TODL = TODL_sub['SP'][0][count]

    z_TODL = z_from_p(P_TODL)
    SA_TODL = gsw.SA_from_SP(SP_TODL, P_TODL, lon, lat)
    CT_TODL = gsw.CT_from_t(SA_TODL, T_TODL, P_TODL)
    sig0_TODL = gsw.sigma0(SA_TODL, CT_TODL)
    oxysat = gsw.O2sol(SA_TODL, CT_TODL, P_TODL, lon, lat)
    
    O2_TODL = O2_mumol / oxysat * 100   #%air saturation
    return O2_TODL, z_TODL, sig0_TODL, O2_mumol

# =============================================================================
#     #TODO: use this for 2019 cruise!
# =============================================================================
def calc_O2conc_interp(path, file, count):
    #for 2019 cruise calculate O2 concentration from O2 in % air saturation
    lon = 20.5
    lat = 57.3
    matlab = ssio.loadmat(path + 'oxy_shift/' + file[:-4] + '_TODL_merged_shift_oxy.mat')
    CTD_sub = matlab['CTD']
    O2sat = CTD_sub['O2'][0][count]
    P_CTD = CTD_sub['P'][0][count]
    # T_CTD = CTD_sub['T'][0][count]
    CT_CTD = CTD_sub['CT'][0][count]
    SA_CTD = CTD_sub['SA'][0][count]
    C100 = gsw.O2sol(SA_CTD, CT_CTD, P_CTD, lon, lat)
    O2conc = O2sat / 100 * C100       #C100 is in units umol/kg!!!
    
    z_CTD = z_from_p(P_CTD)
    # CT_CTD = gsw.CT_from_t(SA_CTD, T_CTD, P_CTD)
    # sig0_CTD = gsw.sigma0(SA_CTD, CT_CTD)
    sig0_CTD_from_matlab = CTD_sub['SIGTH'][0][count]
    return O2conc, z_CTD, sig0_CTD_from_matlab


def calc_O2conc(O2sat, P, SA, CT, lon, lat):
    # O2conc = O2sat.copy()
    lon_array = np.full((len(P), len(O2sat[0])), lon)
    lat_array = np.full((len(P), len(O2sat[0])), lat)
    
    #in units of mumol / kg
    # O2conc = O2sat/100 * gsw.O2sol(SA, CT, P, lon_array, lat_array)
    
    #in units of mumol / l 
    # O2conc = O2conc[:,count] * 1000 / gsw.rho(SA[:,count], CT[:,count], P[:,count])
    O2conc = O2sat/100 * gsw.O2sol(SA, CT, P, lon_array, lat_array) * 1000 / gsw.rho(SA, CT, P)
    
    return O2conc

def p100O2(p_atm, T):
    #pH2O - saturated water vapor pressure at temperature T
    #p_atm - actual barometric pressure
    #T - actual temperature
    pH2O = 6.112 * np.exp(17.62 * T/(243.12 + T))
    return 0.2095* (p_atm - pH2O)

def calc_air_sat(p_atm, T, pO2):
    air_sat = 100 * pO2 / p100O2(p_atm, T)
    return air_sat

#vectorized version (acts on every element of an array)
def Skif(Reb):
    """
    vectorized version of the Shih(2005) turbulence parametrizaion
    #calculates mixing efficiency
    """

    def basic_Skif(Reb):
        if Reb < 7:
            return 0
            
        elif ((Reb >= 7) and (Reb<=100)):
            return 0.2
            
        else:
            return 2*Reb**(-0.5)
            
    vSkif = np.vectorize(basic_Skif, otypes=[float]) 
    
    return vSkif(Reb)

def get_turbulent_diffusivity_Shih(Reb_number,eps,N_squared):    
    Gamma_Skif = Skif(Reb_number)
    turbulent_diffusivity_Skif = Gamma_Skif * eps / (N_squared)
    #remove negative diffusivity
    turbulent_diffusivity_Skif[turbulent_diffusivity_Skif<0] = np.nan
    return turbulent_diffusivity_Skif

# =============================================================================
# # for v geostrophic calculation
# =============================================================================

def calc_cumtrapz(dv_dz, z, start = 0):
    #numerically integrates dv_dz cumulative from the index start; z_new is centered betweeen old depths
        start = int(start)
        z_new = z[start:]
        #integrate dv_dz from surface or mixed depth to yield v; replace nans with 0 for integration:
        v_from_rho = np.zeros((len(z_new), len(dv_dz[0])))
        for i in range(len(dv_dz[0])): #loop over profiles
            dv_dz_col = np.nan_to_num(dv_dz[start:,i])
            v_from_rho[:,i] = cumulative_trapezoid(dv_dz_col, x = z_new, initial = 0)
            
        return z_new, v_from_rho

# =============================================================================
# #for wind data
# =============================================================================

def get_wind_model_data(path_to_wind, first, second, third):
    if first:
        year_number = 2017
        winddata_u = np.genfromtxt(path_to_wind + '2017_u.dump', comments = '%', skip_header=4)
        winddata_v = np.genfromtxt(path_to_wind + '2017_v.dump', comments = '%', skip_header=4)
    if second:
        year_number = 2018
        winddata_u = np.genfromtxt(path_to_wind + '2018_u.dump', comments = '%', skip_header=4)
        winddata_v = np.genfromtxt(path_to_wind + '2018_v.dump', comments = '%', skip_header=4)
    if third:
        year_number = 2019
        winddata_u = np.genfromtxt(path_to_wind + '2019_u.dump', comments = '%', skip_header=4)
        winddata_v = np.genfromtxt(path_to_wind + '2019_v.dump', comments = '%', skip_header=4)

    tu_wind = winddata_u[0:, 0]
    u_wind = winddata_u[0:, 1]
    v_wind = winddata_v[0:, 1]
    
    t2_wind = np.copy(tu_wind)
    tOffset = -secondsUntil(year_number)+secondsUntil(1970)           #to convert to epoch time
    
    t2_wind[:] = tu_wind[:] - tOffset
    t_wind = np.arange(0, len(t2_wind)).astype(datetime)
    for count, item in enumerate(t2_wind):
        t_wind[count] = datetime.utcfromtimestamp(item)
    
    return t_wind, u_wind, v_wind

# =============================================================================
# #for ADCP data
# =============================================================================

def getADCP_plot_definitions(first, second, third, deep):
    offset_adcp = 0
    offset_adcp_down = 0
    #old: 0, 0, 2, 0
    if first == True:
        monthLow = 10
        monthHigh = 10
        T_bounds = [4, 13.5]
        if deep == True:
            depth_water = 100
            dayLow = 19
            hourLow = 12
            minuteLow = 0
            secondLow = 0
            dayHigh = 23
            hourHigh = 16
            minuteHigh = 0
            secondHigh = 0
            
            ylow = -90
            yhigh = -0
            
            offset_adcp = +1
            
        else:
            depth_water = 75
            dayLow = 20
            hourLow = 6
            minuteLow = 0
            secondLow = 0
            dayHigh = 24
            hourHigh = 16
            minuteHigh = 0
            secondHigh = 0

            ylow = -75
            yhigh = -55

    elif second == True:
        T_bounds = [2, 11.5]
        if deep == True:   
            depth_water = 100
            monthLow = 3
            dayLow = 3
            hourLow = 15
            minuteLow = 30
            secondLow = 0
            monthHigh = 3
            dayHigh = 7
            hourHigh = 5
            minuteHigh = 30
            secondHigh = 0
            
            ylow = -92.4836
            yhigh = -0
            
            offset_adcp = 0    
            
        else:  
            depth_water = 65
            offset_adcp_down = 3
            monthLow = 2
            dayLow = 26
            hourLow = 12
            minuteLow = 0
            secondLow = 0
            monthHigh = 3
            dayHigh = 7
            hourHigh = 12
            minuteHigh = 0
            secondHigh = 0
            
            ylow = -65
            yhigh = -0
            
    elif third == True:
        monthLow = 7
        dayLow = 11
        hourLow = 6
        T_bounds = [4, 17.5]
        if deep == True:
            depth_water = 99
            minuteLow = 0
            secondLow = 0
            monthHigh = 7
            dayHigh = 22
            hourHigh = 15
            minuteHigh = 0
            secondHigh = 0
            
            ylow = -94.0757
            yhigh = -0
            offset_adcp = +0    
        else:
            depth_water = 78
            minuteLow = 0
            secondLow = 0
            monthHigh = 7
            dayHigh = 22
            hourHigh = 16
            minuteHigh = 0
            secondHigh = 0
            
            ylow = -78
            yhigh = -0
            
            offset_adcp = 0         
            offset_adcp_down = 1
    return offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh

# =============================================================================
# #adcp cropping
# =============================================================================
def checkRows(array):
    verybadRows = 0
    for row_index in range(len(array[0])):       #for every timeline in one depth
        row = array[0:, row_index]              #isolate timeline
        if (np.isnan(row).all(axis=0)) == True:
            verybadRows += 1
            print('very bad row at index ', row_index)
    print('total number of bad rows: ', verybadRows)

def deleteRows(depth, array, cutoff_start, cutoff_end):
    verybadRows = 0
    for row_index in range(len(array[0])):       #for every timeline in one depth
        row = array[0:, row_index]              #isolate timeline
        if (np.isnan(row).all(axis=0)) == True:
            verybadRows += 1
            print('very bad row at index ', row_index)
    depth_rows = depth[cutoff_start:(len(array[0])-cutoff_end)]
    array_rows = array[0:, cutoff_start:(len(array[0])-cutoff_end)]
    return depth_rows, array_rows

def deleteCols(time, array, cutoff_start, cutoff_end):
    time_cols = time[cutoff_start:(len(time)-cutoff_end)]
    array_cols = array[cutoff_start:(len(array)-cutoff_end), 0:]
    return time_cols, array_cols

def crop2d(array, cutoff_rows, cutoff_cols, down):
    if down == False:
        array_rows = array[0:, cutoff_rows[0]:(len(array[0])-cutoff_rows[1])]
        array_cols = array_rows[cutoff_cols[0]:(len(array)-cutoff_cols[1]), 0:]
    else:
        array_rows = array[0:, cutoff_rows[2]:(len(array[0])-cutoff_rows[3])]
        array_cols = array_rows[cutoff_cols[2]:(len(array)-cutoff_cols[3]), 0:]
    return array_cols

def crop1d(array, cutoffs, down):
    if down == False:
        array_out = array[cutoffs[0]:(len(array)-cutoffs[1])]
    else:
        array_out = array[cutoffs[2]:(len(array)-cutoffs[3])]
    return array_out

# =============================================================================
# #adcp interpolating
# =============================================================================
def interpolateLin(t, array):
    for zval in range(len(array[0])):       #for every timeline in one depth
        data_nan = array[0:, zval]              #isolate timeline
        ind_good = ~np.isnan(data_nan)
        data_good = np.interp(t, t[ind_good], data_nan[ind_good], left = 0, right = 0)  #left -> values left from the first datapoint x_l, default: f(x_l)
        if zval == 0:
            data_out = data_good
            # print('v_out', v_out, len(v_out))
        else: 
            data_out = np.vstack([data_out, data_good])
            # v_out = np.c_[v_out, temp_out]
            # print('v_out', v_out, len(v_out))
    return data_out.T

# =============================================================================
# filtering with butterworth filter
# ============================================================================= 
def butterThis(array, f_sampling, f_cutoff, lowOrHigh, order):      
    #input array 2d, e.g. currx or curry, output already transposed
    norm = f_cutoff / (f_sampling / 2)
    b, a = signal.butter(order, norm, lowOrHigh)
    for zval in range(len(array[0])):       #for every timeline in one depth
        temp = array[0:, zval]              #isolate timeline
        #check for nan values and replace:
        # temp = interpolateLin(t, temp)
        # print(temp)
        temp_out = signal.filtfilt(b, a, temp) #filter timeline
        # print('temp_out', temp_out, len(temp_out))
        if zval == 0:
            v_out = temp_out
            # print('v_out', v_out, len(v_out))
        else: 
            v_out = np.vstack([v_out, temp_out])
            # v_out = np.c_[v_out, temp_out]
            # print('v_out', v_out, len(v_out))
    return v_out

# =============================================================================
# #calculate shear
# =============================================================================
def calcShear(current_x, current_y, depth_adcp):
    #input current matrices and depth_adcp as array
    #returns shear matrix
    shear_adcp = current_x.copy()
    for indexADCP, u_adcp in enumerate(current_x):
        du_dz = -np.gradient(current_x[indexADCP], depth_adcp)
        dv_dz = -np.gradient(current_y[indexADCP], depth_adcp)
        shear_squared = du_dz[:]**2 + dv_dz[:]**2
        shear_adcp[indexADCP] = shear_squared
    return shear_adcp

# =============================================================================
# #spectra
# =============================================================================

def calcSpectrumRotary(array, fs, npersegment):
    for zval in range(len(array[0])):       #for every timeline in one depth
        
        f, ps = signal.welch(array[:,zval], fs, nperseg=npersegment)#len(array[:,0])/10)
        if zval == 0:
            data_out = ps
        else: 
            data_out[:] = data_out[:] + ps[:]
    return f, data_out/len(array[0])

def calcSpectrumWelch(array, fs, npersegment):
    for zval in range(len(array[0])):       #for every timeline in one depth
        f, Pxx_den = signal.welch(array[:,zval], fs, nperseg=npersegment)
        # print(len(array[:,zval]), array[:,zval])
        # print(f)
        if zval == 0:
            data_out = Pxx_den
        else: 
            data_out[:] = data_out[:] + Pxx_den[:]
    return f, data_out

def calcSpectrumWelchColor(array, fs, npersegment):
    f, Pxx_den = signal.welch(array[:,0], fs, nperseg=npersegment)
    # result = np.zeros((len(array[0]), len(f)))
    result = np.zeros((len(f), len(array[0])))
    for zval in range(len(array[0])):       #for every timeline in one depth
        f, Pxx_den = signal.welch(array[:,zval], fs, nperseg=npersegment)
        result[:,zval] = Pxx_den
    return f, result
