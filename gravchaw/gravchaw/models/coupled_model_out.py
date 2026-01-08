"""
This module is the coupled hydrogravimetric model, linking flopy to `gravi4GW-hybrid` module.
It calculates time-lapse gravity (TLG) data at observation points and writes the output to an ascii file
in the working directory. 
In a case of joint inversion with hydrogeological observations, model output includes TLG
and hydrogeological data files in ascii formats. 
"""
import os
import flopy
import pandas as pd

from gravchaw.models.gravi4gw_hybrid import tlg_hybrid

def coupledmodel_out(grav_output_name,
                      len_grav_obs,
                      reference_time,
                      target_time,
                      x_gravstn,
                      y_gravstn,
                      z_gravstn,
                      station_name,
                      ws):
    """
    Contains Gravi4GW-hybrid and links it to flopy.
    
       Arguments:
       ----------
       grav_output_name : str
                        The name of TLG output file, which must be the same as the name of TLG observation file.
       
       len_grav_obs : int
                    Number of time intervals to model TLG.
                    
       reference_time : list of int(s)
                      A list of time step index (or indices) to extract reference heads.
                      
       target_time : list of int(s) 
                   A list of time step index (or indices) to extract target heads.   
                   
       station_name : List of str(s)
                    A list of gravity station names.  
                    
       x_gravstn : list of float or int
                 A list of x coordinates (in meter) of gravity stations.
                 
       y_gravstn : list of float(s) or int(s)
                 A list of y coordinates (in meter) of gravity stations.   
       
       z_gravstn : list of float or int
                 A list of z coordinates (in meter) of gravity stations. 
                 
       ws :  str
          A path to pyemu working directory.
          
       Returns:
       -------
       ascii_file_path : str
                       Path to ascii file containing TlG output (and hydrogeological outputs in a case of joint inversion).
      
     *** Note:
             TLG data unit is microGal. The unit of hydrogeological data is sepcified during creating the model
             through flopy.
    
    """
    
    # load the groundwater model to link flopy and Gravi4GW-hybrid
    sim = flopy.mf6.MFSimulation.load(sim_ws=ws, exe_name='./mf6')
    sim.write_simulation()
    sim.run_simulation() # simulate hydraulic heads
    gwf = sim.get_model()
    modelgrid_m = gwf.modelgrid # to extract model's spatial grid
    dis=gwf.dis
    sim_unit=dis.length_units.get_data() # to extract model's unit
    hds = gwf.output.head()     # to extract htdraulic heads 
    sto_m = gwf.sto             # to extract porosity
    extracte_times = hds.get_times()
    porosity = sto_m.sy._get_data()
    # calculate TLG
    delta_g = {}
    for itime in range(0, len_grav_obs):
        # reference and target heads extracted based on provided time indecies
        ext_refe_time=extracte_times[reference_time[itime]] 
        reference_head = hds.get_data(totim=ext_refe_time) 
        ext_targ_time=extracte_times[target_time[itime]]
        head = hds.get_data(totim=ext_targ_time)
        model_out = tlg_hybrid(modelgrid_m,
                                    reference_head,
                                    head,
                                    x_gravstn,
                                    y_gravstn,
                                    z_gravstn,
                                    porosity,
                                    sim_unit) 
        delta_g[f'{ext_refe_time:03}-{ext_targ_time:03}'] = model_out['gravity']
    # extract model_out as an ascii file   
    df = pd.DataFrame.from_dict(delta_g, orient='index')    
    df.columns = station_name
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)   
    grav_path = os.path.join(ws, grav_output_name)
    df.to_csv(grav_path, float_format="%0.10f", index=False)    
