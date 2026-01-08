import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyemu

# function added thru PstFrom.add_py_function()
def tlg_gw(grav_output_name,
           len_grav_obs,
           reference_time,
           target_time,
           station_name,
           x_gravstn,
           y_gravstn,
           z_gravstn,
           ws):

    from gravchaw.models.coupled_model_out import coupledmodel_out
    
    coupledmodel_out(grav_output_name,
                          len_grav_obs,
                          reference_time,
                          target_time,
                          x_gravstn,
                          y_gravstn,
                          z_gravstn,
                          station_name,
                          ws) 
def main():

    try:
       os.remove(r'heads.csv')
    except Exception as e:
       print(r'error removing tmp file:heads.csv')
    try:
       os.remove(r'grav.csv')
    except Exception as e:
       print(r'error removing tmp file:grav.csv')
    pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv',chunk_len=50)
    tlg_gw(grav_output_name='grav.csv', len_grav_obs=24, reference_time=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target_time=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], station_name=['G000', 'G001', 'G002', 'G003', 'G004', 'G005', 'G006', 'G007', 'G008', 'G009'], x_gravstn=[728.4138455687, 880.5629516927, 710.3008567444, 572.6421416798, 79.9688456592, 108.9496277781, 224.8727562535, 275.5891249615, 833.4691807495, 626.9811081527], y_gravstn=[1924.0703508486, 1782.7890380192, 1427.7744570632, 877.1395968049, 1474.8682280063, 1119.8536470503, 732.2356862106, 670.651524208, 228.6945968954, 127.2618594794], z_gravstn=[66.0313210424, 64.6444932239, 62.7723216961, 64.8550356475, 90.2815125659, 86.9680833282, 72.1098325986, 69.5567769695, 59.6112512619, 55.0933101839], ws='.')

if __name__ == '__main__':
    mp.freeze_support()
    main()

