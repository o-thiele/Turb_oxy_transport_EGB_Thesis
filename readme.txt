####################################################
Code structrue:
The code is provided in different files, each named roughly after what they're doing. First, the files starting with 00 have to be run, then 01, 02 ... 99.
77 are plot files
99 are files for debugging.    

Most files are structured in a main, that can be called for each cruise individually. For plotting this is sometimes required, because axes are not cleared accordingly.

####################################################
File structrue:
Right now, the code uses relative paths. This can easily be adapted by manipulating the function "getPaths" in own_functions.py. Almost all scripts make use of own_functions.py and/or metadata_mooring.pkl, which is created via metadata.py.
When using relative paths, the rough structure is as follows:

parent_folder/
    code/

    data_adcp_and_moorings/
        EMB169/
            TC_deep/
                adcp/
                    data/
                    proc/
                    rawdata/
            TC_shallow/
        EMB177/
            --as for EMB169--
        EMB217/
            TC_deep/
                --as for EMB169/TC_deep/
            TC_shallow/
                adcp/
                    data/
                ADCP600/
                    data/
                    proc/
                    rawdata/
                ADCP1200/
                    data/
                    proc/
                    rawdata/

    data_mss/

    winddata_meteo/


Here code contains all the python scripts. 
The ADCP data is to be placed in the data_adcp_and_moorings folder, in the corresponding subdirectory, with the .mat file in the corresponding data/ folder. For 2019 the .mat files for the 2019 cruise have to be in the data/ folder as well, but originally were in the ADCP600/data/ and ADCP1200/data/ folders.
The MSS data parent folder as downloaded from "http://doi.io-warnemuende.de/10.12754/data-2022-0002" is to be renamed as "data_mss".
The winddata_meteo/ folder contains the model wind data

####################################################
individual files and their function:

00_adcp_crop-and-interpolate.py
    -crops and interpolates adcp data and saves them in the data/ folders in a new subfolder npz/

01_adcp_z-avg_shallow_down_2019.py
    -averages the downward facing 2019 adcp to 1m bins

02_adcp_filter.py
    -filters the adcp data in low and high frequencies

03_mss_interpolate-data.py
    -interpolates the mss data; most plots are commented out, just o2 as an example is active; data and figures are saved to a new folder results/mss/mss_interpolated with subsequent cruise folders in the parent directory

04_mss_calculate_fluxes_and_Ri.py:
    calculates fluxes and Richardson numbers for the unfiltered and filtered data and saves them to results/fluxes_and_Ri/ etc.; plot saving is deactivated

05_calculate_v_geostrophic.py:
    calculates geostrohic velocity and comparison cuts and plots them in results/plots/v_geo/ + cruise specs

06_adcp_plot_spectra.py:
    calculates and plots spectra in results/plots/adcp/spectra/ etc

77 .. plots what is stated in the name mostly..
