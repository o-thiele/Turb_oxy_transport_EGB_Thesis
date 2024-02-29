import numpy as np
import pylab as pl
import os
import matplotlib.dates as mdates
import scipy.io as ssio
from datetime import datetime
import time as timing
#own functions
import own_functions as own

def main(first, second, third, deep):
    brutal_cut = True
    print('currently processing: ', first, second, third, deep)
    path_to_mss_files, path_mss_interpolated, cruise, mooring, path_to_wind, path_ADCP_data, path_results, operating_system = own.getPaths('relative', first, second, third, deep)
    
    if first == True:
        year = 2017
        if deep == True:
            cutoff_rows = [0, 14, 0, 0]     #start, end, down_start, down_end
            cutoff_cols = [0, 0, 0, 0]      #start, end, down_start, down_end
            fileADCP = 'emb169_tc1_ADCP_val'
            
        else: 
            cutoff_rows = [0, 9, 0, 0]     #start, end, down_start, down_end
            cutoff_cols = [0, 0, 0, 0]      #start, end, down_start, down_end
            fileADCP = 'emb169_tc2_ADCP_val'

    elif second == True:
        year = 2018
        if deep == True:
            cutoff_rows = [0, 18, 0, 0]     #start, end, down_start, down_end
            cutoff_cols = [200, 620, 0, 0]      #start, end, down_start, down_end   #    old: 62, 430, 0, 0
            fileADCP = 'EMB177_TC-tief_adcp_val'

        else:
            cutoff_rows = [0, 27, 0, 9]     #start, end, down_start, down_end
            cutoff_cols = [100, 352, 100, 319]      #start, end, down_start, down_end
            fileADCP = 'EMB177_TC-flach_adcp-up_val'
            fileADCP_down = 'EMB177_TC-flach_adcp-down_val'

    elif third == True:
        # brutal_cut = False
        year = 2019
        if deep == True:
            cutoff_rows = [0, 13, 0, 0]     #start, end, down_start, down_end
            if brutal_cut:
                cutoff_rows = [0, 20, 0, 0]     #start, end, down_start, down_end
            cutoff_cols = [100, 140, 0, 0]      #start - old: 2, end, down_start, down_end
            fileADCP = 'EMB217_TC-tief_adcp300_val'

        else:
            cutoff_rows = [0, 0, 0, 22]     #start, end, down_start, down_end
            if brutal_cut:
                cutoff_rows = [0, 16, 0, 22]     #start, end, down_start, down_end
            cutoff_cols = [0, 200, 0, 0]      #start, end - old: 0, down_start, down_end
            fileADCP = 'EMB217_TC-flach_adcp600_val'
            fileADCP_down = 'EMB217_TC-flach_adcp1200_val'

            
    path_results_temp = path_ADCP_data + 'npz/'
    # path_figures = path_results + 'adcp/'
        
    # finepathSB = '../' + cruise + mooring + 'Seabird/data/'
    # finepathRBR = '../' + cruise + mooring + 'RBR/data/'
    # finepathPME = '../' + cruise + mooring + 'PME/'
    # pathADCP = finepathADCP + 

    if not os.path.exists(path_results_temp) == True:
        os.makedirs(path_results_temp)
    
    # =============================================================================
    # for ADCP data
    # =============================================================================    
    wholeFile = ssio.loadmat(path_ADCP_data + fileADCP)
    data = wholeFile['adcpavg']
    # print(data)
    
    time = data[0, 0]['rtc']    #matlab time
    t3 = np.copy(time)
    t3Offset = own.secondsUntil(1970)           #to convert to epoch time
    t3[:] = time[:] * 24 * 60 *60 - t3Offset - 24 * 60 *60 #why 1 day off?
    t_matlab = np.arange(0, len(t3)).astype(float)
    t = np.arange(0, len(t3)).astype(datetime)
    for count, item in enumerate(t3[:,0]):
        t[count] = datetime.utcfromtimestamp(item)
        t_matlab[count] = time[count][0]    
    
    depth_temp = data[0, 0]['depth'][0]
    depth = np.copy(depth_temp)
    depth[:] = -1 * depth_temp[:]
    
    mdepth = data[0,0]['mdepth'][0][0] #mounting depth
    p = data[0,0]['press']
    # print(p)
    # p[:] = -p[:]
    
    curr = data[0,0]['curr']      #currents
    currx = curr.real
    curry = curr.imag
    # abcs = data[0,0]['abcs']
    
    # abs1 = np.zeros((len(abcs), len(abcs[0,:])), dtype=float)
    # abs2 = abs1.copy()
    # abs3 = abs1.copy()
    # abs4 = abs1.copy()
    # absM = abs1.copy()
    # for count in range((len(abcs))):
    #     abs1[count,:] = abcs[count,:][:,0]
    #     abs2[count,:] = abcs[count,:][:,1]
    #     abs3[count,:] = abcs[count,:][:,2]
    #     abs4[count,:] = abcs[count,:][:,3]
    
    # abs1 = np.log10(abs1)
    # abs2 = np.log10(abs2)
    # abs3 = np.log10(abs3)
    # abs4 = np.log10(abs4)
    # absM = np.mean(np.array([abs1, abs2, abs3, abs4]), axis=0 )
    
# =============================================================================
#     print('saving upward adcp')
#     np.savez_compressed(path_results_temp + 'adcp_raw_', t_matlab = t_matlab, t = t, depth = depth, mdepth = mdepth, p = p, currx = currx, curry = curry)
# =============================================================================
    
    # np.save(pathResults + 'adcp/data/npy/time_matlab', t_matlab)
    # np.save(pathResults + 'adcp/data/npy/time', t)
    # np.save(pathResults + 'adcp/data/npy/depth', depth)
    # np.save(pathResults + 'adcp/data/npy/mdepth', mdepth)
    # np.save(pathResults + 'adcp/data/npy/pressure', p)
    # np.save(pathResults + 'adcp/data/npy/currx', currx)
    # np.save(pathResults + 'adcp/data/npy/curry', curry)
    # np.save(pathResults + 'adcp/data/npy/abcs_mean', absM)

# =============================================================================
#     #if downward adcp exists -> also process
# =============================================================================
    try:
        wholeFile = ssio.loadmat(path_ADCP_data + fileADCP_down)
        data = wholeFile['adcpavg']
        
        time = data[0, 0]['rtc']    #matlab time
        t2 = np.copy(time)
        t2Offset = own.secondsUntil(1970)           #to convert to epoch time
        t2[:] = time[:] * 24 * 60 *60 - t2Offset - 24 * 60 *60 #why 1 day off?
        t_matlab_down = np.arange(0, len(t2)).astype(float)
        t_down = np.arange(0, len(t2)).astype(datetime)
        for count, item in enumerate(t2[:,0]):
            t_down[count] = datetime.utcfromtimestamp(item)
            t_matlab_down[count] = time[count][0]
      
        depth_temp = data[0, 0]['depth'][0]
        depth_down = np.copy(depth_temp)
        depth_down[:] = -1 * depth_temp[:]
        
        mdepth_down = data[0,0]['mdepth'][0][0] #mounting depth
        p_down = data[0,0]['press']
        # p_down[:] = -p_down[:]
        
        # w_down = data[0,0]['vu']
        curr_down = data[0,0]['curr']      #currents
        currx_down = curr_down.real
        curry_down = curr_down.imag
        # abcs_down = data[0,0]['abcs']
        
        # abs1_down = np.zeros((len(abcs_down), len(abcs_down[0,:])), dtype=float)
        # abs2_down = abs1_down.copy()
        # abs3_down = abs1_down.copy()
        # abs4_down = abs1_down.copy()
        # absM_down = abs1_down.copy()
        # for count in range((len(abcs_down))):
        #     abs1_down[count,:] = abcs_down[count,:][:,0]
        #     abs2_down[count,:] = abcs_down[count,:][:,1]
        #     abs3_down[count,:] = abcs_down[count,:][:,2]
        #     abs4_down[count,:] = abcs_down[count,:][:,3]
        
        # abs1_down = np.log10(abs1_down)
        # abs2_down = np.log10(abs2_down)
        # abs3_down = np.log10(abs3_down)
        # abs4_down = np.log10(abs4_down)
        # absM_down = np.mean(np.array([abs1_down, abs2_down, abs3_down, abs4_down]), axis=0)
        
# =============================================================================
#         print('saving downward adcp')
# 
#         np.savez_compressed(path_results_temp + 'adcp_raw_down', t_matlab = t_matlab_down, t = t_down,
#                             depth = depth_down, mdepth = mdepth_down, p = p_down,
#                             currx = currx_down, curry = curry_down)
# =============================================================================

        
        # np.save(pathResults + 'adcp/data/npy/time_matlab_down', t_matlab_down)
        # np.save(pathResults + 'adcp/data/npy/time_down', t_down)
        # np.save(pathResults + 'adcp/data/npy/depth_down', depth_down)
        # np.save(pathResults + 'adcp/data/npy/mdepth_down', mdepth_down)
        # np.save(pathResults + 'adcp/data/npy/pressure_down', p_down)
        # np.save(pathResults + 'adcp/data/npy/currx_down', currx_down)
        # np.save(pathResults + 'adcp/data/npy/curry_down', curry_down)
        # np.save(pathResults + 'adcp/data/npy/abcs_mean_down', absM_down)

    except:
        print('no downward facing ADCP data found')
        pass
    
    # =============================================================================
    #     #cropping
    # =============================================================================
    #crop rows
    depthx_rows, currx_rows = own.deleteRows(depth, currx, cutoff_rows[0], cutoff_rows[1])
    depthy_rows, curry_rows = own.deleteRows(depth, curry, cutoff_rows[0], cutoff_rows[1])
    #crop columns
    timex_cols, currx_cols = own.deleteCols(t, currx_rows, cutoff_cols[0], cutoff_cols[1])
    timey_cols, curry_cols = own.deleteCols(t, curry_rows, cutoff_cols[0], cutoff_cols[1])
    
    t_matlab = own.crop1d(t_matlab, cutoff_cols, False)
    p = own.crop1d(p, cutoff_cols, False)
    # w = own.crop2d(w, cutoff_rows, cutoff_cols, False)
    # absM = own.crop2d(absM, cutoff_rows, cutoff_cols, False)
    own.checkRows(currx_cols)
    own.checkRows(curry_cols)
    
    np.savez_compressed(path_results_temp + 'adcp_cropped', t_matlab = t_matlab, t = timex_cols,
                        depth = depthx_rows, mdepth = mdepth, p = p,
                        currx = currx_cols, curry = curry_cols)
    
    try:
        depthx_rows_down, currx_rows_down = own.deleteRows(depth_down, currx_down, cutoff_rows[2], cutoff_rows[3])
        depthy_rows_down, curry_rows_down = own.deleteRows(depth_down, curry_down, cutoff_rows[2], cutoff_rows[3])
        timex_cols_down, currx_cols_down = own.deleteCols(t_down, currx_rows_down, cutoff_cols[2], cutoff_cols[3])
        timey_cols_down, curry_cols_down = own.deleteCols(t_down, curry_rows_down, cutoff_cols[2], cutoff_cols[3])
    
        t_matlab_down = own.crop1d(t_matlab_down, cutoff_cols, True)
        p_down = own.crop1d(p_down, cutoff_cols, True)
        # w_down = own.crop2d(w_down, cutoff_rows, cutoff_cols, True)
        # absM_down = own.crop2d(absM_down, cutoff_rows, cutoff_cols, True)
        own.checkRows(currx_cols_down)
        own.checkRows(curry_cols_down)
        
        np.savez_compressed(path_results_temp + 'adcp_cropped_down', t_matlab = t_matlab_down, t = timex_cols_down,
                            depth = depthx_rows_down, mdepth = mdepth_down, p = p_down,
                            currx = currx_cols_down, curry = curry_cols_down)

    except:
        pass
    
# =============================================================================
#     #interpolate
# =============================================================================
    currx_interp = own.interpolateLin(t_matlab, currx_cols)
    curry_interp = own.interpolateLin(t_matlab, curry_cols)
    
    if third and deep and brutal_cut == False:
        #just for plotting
        
        #remove deep mooring        15-.Jul - 1200  until 17. Jul 0000 rows 0-7 
        #                   and:    start           until 12. Jul 0600 rows 0-5
        # due to large lacks in data
        
        #for all rows
        for i in range(len(currx_interp[0])):
            for count, value in enumerate(currx_interp[:,i]):
                if ((timex_cols[count] < datetime(2019,7,12,6)) and ((i+6) > len(currx_interp[0]))):
                    currx_interp[count, i] = np.nan
                    curry_interp[count, i] = np.nan
                elif ((timex_cols[count] < datetime(2019,7,17)) and (timex_cols[count] > datetime(2019,7,15,12)) and ((i+8) > len(currx_interp[0]))):
                    currx_interp[count, i] = np.nan
                    curry_interp[count, i] = np.nan
        
        #or the "hard way" for spectra calculation and don't care about nans: remove 7 rows more
            
    if third and deep == False and brutal_cut == False:
        #remove shallow mooring     15-.Jul - 1200  until 17. Jul 0000 rows 0-15 
        #                   and:    start           until 12. Jul 0600 rows 0-9
        #                   and:    17. Jul - 0000  until 18. Jul 1200 rows 0-9
        # due to large lacks in data
        
        #for all rows
        for i in range(len(currx_interp[0])):
            for count, value in enumerate(currx_interp[:,i]):
                if ((timex_cols[count] < datetime(2019,7,12,6)) and ((i+9) > len(currx_interp[0]))):
                    currx_interp[count, i] = np.nan
                    curry_interp[count, i] = np.nan
                elif ((timex_cols[count] < datetime(2019,7,17)) and (timex_cols[count] > datetime(2019,7,15,12)) and ((i+16) > len(currx_interp[0]))):
                    currx_interp[count, i] = np.nan
                    curry_interp[count, i] = np.nan
                elif ((timex_cols[count] > datetime(2019,7,17)) and (timex_cols[count] < datetime(2019,7,18,12)) and ((i+9) > len(currx_interp[0]))):
                    currx_interp[count, i] = np.nan
                    curry_interp[count, i] = np.nan
        
        #or the "hard way" for spectra calculation and don't care about nans: remove 15 rows more

    if brutal_cut == True:
        print('saving interpolated data..')
        np.savez_compressed(path_results_temp + 'adcp_interpolated', t_matlab = t_matlab, t = timex_cols,
                            depth = depthx_rows, mdepth = mdepth, p = p,
                            currx = currx_interp, curry = curry_interp)
        
        #downwards:
        try:
            currx_interp_down = own.interpolateLin(t_matlab_down, currx_cols_down)
            curry_interp_down = own.interpolateLin(t_matlab_down, curry_cols_down)
            np.savez_compressed(path_results_temp + 'adcp_interpolated_down', t_matlab = t_matlab_down, t = timex_cols_down,
                                depth = depthx_rows_down, mdepth = mdepth_down, p = p_down,
                                currx = currx_interp_down, curry = curry_interp_down)
        except:
            pass
    else:
        print('saving interpolated data.. fine cut for plot')
        np.savez_compressed(path_results_temp + 'adcp_interpolated_fine', t_matlab = t_matlab, t = timex_cols,
                            depth = depthx_rows, mdepth = mdepth, p = p,
                            currx = currx_interp, curry = curry_interp)
        
        #downwards:
        try:
            currx_interp_down = own.interpolateLin(t_matlab_down, currx_cols_down)
            curry_interp_down = own.interpolateLin(t_matlab_down, curry_cols_down)
            np.savez_compressed(path_results_temp + 'adcp_interpolated_fine_down', t_matlab = t_matlab_down, t = timex_cols_down,
                                depth = depthx_rows_down, mdepth = mdepth_down, p = p_down,
                                currx = currx_interp_down, curry = curry_interp_down)
        except:
            pass
    

# =============================================================================
#     #plots
# =============================================================================
#veryt time inefficient, better load again
# =============================================================================
#     offset_adcp, offset_adcp_down, depth_water, T_bounds, monthLow, dayLow, hourLow, minuteLow, secondLow, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh, ylow, yhigh = own.getADCP_plot_definitions(first, second, third, deep)
#     ylow = ylow -3
#     hourLow = hourLow -1
#     hourHigh = hourHigh +1
#     largeTicks = mdates.DayLocator()
#     smallTicks = mdates.HourLocator(byhour=[6, 12, 18])
#     dayformat = mdates.DateFormatter('%d.%b')
#     
#     fig = pl.figure(1, figsize =(22,16))
#     pl.clf()
#     fig.suptitle(mooring + ' --- original, cropped and interpolated current velocities', fontsize=14)
#     rows = 4
#     cols = 2
#     
#     pl.subplot(rows,cols,1, adjustable='box')
#     pl.title('original')
#     pl.pcolormesh(t,depth,currx.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(t_down,depth_down,currx_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     # pl.plot(t,p, ls = '--', color = 'black', lw = 0.7)
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('u [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
#     
#         
#     pl.subplot(rows,cols,3, adjustable='box')
#     pl.title('rows removed')
#     pl.pcolormesh(t,depthx_rows,currx_rows.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(t_down,depthx_rows_down,currx_rows_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     # print(len(p), len(t), t, p)
#     # pl.plot(t,p, ls = '--', color = 'black', lw = 0.7)
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('u [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
#             
#     pl.subplot(rows,cols,5, adjustable='box')
#     pl.title('cols removed')
#     pl.pcolormesh(timex_cols,depthx_rows,currx_cols.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(timex_cols_down,depthx_rows_down,currx_cols_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     pl.plot(timex_cols,-p, ls = '--', color = 'black', lw = 0.7)
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('u [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
# 
#     pl.subplot(rows,cols,7, adjustable='box')
#     pl.title('interpolated')
#     pl.pcolormesh(timex_cols,depthx_rows,currx_interp.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(timex_cols_down,depthx_rows_down,currx_interp_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     pl.plot(timex_cols,-p, ls = '--', color = 'black', lw = 0.7)
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('u [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
# 
# 
#     pl.subplot(rows,cols,2, adjustable='box')
#     pl.pcolormesh(t,depth,curry.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(t_down,depth_down,curry_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('v [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
# 
#     pl.subplot(rows,cols,4, adjustable='box')
#     pl.pcolormesh(t,depthy_rows,curry_rows.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(t_down,depthy_rows_down,curry_rows_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     # pl.plot(t,p, ls = '--', color = 'black', lw = 0.7)
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('u [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
# 
#     pl.subplot(rows,cols,6, adjustable='box')
#     pl.pcolormesh(timey_cols,depthy_rows,curry_cols.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(timey_cols_down,depthy_rows_down,curry_cols_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     pl.plot(timex_cols,-p, ls = '--', color = 'black', lw = 0.7, label = '-p')
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('u [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
#     pl.legend()
#     
#     pl.subplot(rows,cols,8, adjustable='box')
#     pl.pcolormesh(timey_cols,depthy_rows,curry_interp.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     try:
#         pl.pcolormesh(timey_cols_down,depthy_rows_down,curry_interp_down.T,cmap=pl.cm.RdBu_r, shading='nearest', vmin=-0.25,vmax=0.25)
#     except:
#         pass
#     pl.plot(timex_cols,-p, ls = '--', color = 'black', lw = 0.7, label = '-p')
#     pl.ylim(ylow, yhigh)
#     pl.ylabel('depth [m]')
#     col = pl.colorbar()
#     col.set_label('u [m/s]')
#     own.matlabTimeConversion(pl, largeTicks, dayformat, smallTicks)
#     ax = pl.gca()
#     ax.set_xlim([mdates.datetime.datetime(year, monthLow, dayLow, hourLow, minuteLow, secondLow), mdates.datetime.datetime(year, monthHigh, dayHigh, hourHigh, minuteHigh, secondHigh)])
#     pl.legend()
#     
#     
#     fig.tight_layout()
#     fig.savefig(path_results_temp + 'adcp_crop.png',dpi=300)
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
