import pickle

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
meta = {'2017': 
            {'depth_deep': 100,
             'mdepth_deep': 93.01,
             'depth_shallow': 75,
             'lat_deep': 57.3197,
             'lon_deep': 20.5989,
             'lat_shallow': 57.3202,
             'lon_shallow': 20.62247,
             'rope_length_deep': 50,
             'highest_sensor_deep': 44.5,
             'rope_length_shallow': 20,
             'highest_sensor_shallow': 10,
             'halocline_center': 7.15,   
             'halocline_interval': 0.85,    #old 7+-0.7
             'adcp_fs_up': 1/900,
             'adcp_fs_down': 1/60,
             'adcp_cutoff_low': 24     #h
             },
        
        '2018': 
            {'depth_deep': 100,
             'mdepth_deep': 92.4836,
             'depth_shallow': 65,
             'lat_deep': 57.3205,
             'lon_deep': 20.5994,
             'lat_shallow': 57.32,
             'lon_shallow': 20.6212,  #two slightly different values in report, chose the value from mooring design
             'rope_length_deep': 25,
             'highest_sensor_deep': 14,
             'rope_length_shallow': 20,
             'highest_sensor_shallow': 10,
             'halocline_center': 6.7,           #old 7.2        6.1-7.3 -> 6.7 +- 0.6
             'halocline_interval': 0.6,         #old 0.75
             'adcp_fs_up': 1/60,
             'adcp_fs_down': 1/60,
             },
        
        '2019': 
            {'depth_deep': 99,
             'mdepth_deep': 94.0757,
             'depth_shallow': 78,
             'mdepth_shallow': 67,
             'lat_deep': 57.32,
             'lon_deep': 20.5976,
             'lat_shallow': 57.3201,
             'lon_shallow': 20.6131, #20.615 in station list, only planned value?
             'rope_length_deep': 30,
             'highest_sensor_deep': 28,
             'rope_length_shallow': 20,
             'highest_sensor_shallow': 15,
             'halocline_center': 7,   #old 7.5            6.3 - 7.7 -> 7 +- 0.7
             'halocline_interval': 0.7,    #old 1.0
             'adcp_fs_up': 1/120, #every 120s
             'adcp_fs_down': 1/120,     #middled such, that every 120s
             'mss_bad_casts': 
                 {'TR1-8_2': [0, 1],
                     'TR1-13': [39,42]
                 }
             }
    }


with open('metadata_moorings.pkl', 'wb') as f:
    pickle.dump(meta, f)
