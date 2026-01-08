"""
This module is the coupled hydrogravimetric model used at the post optimization stage.
It calculates hydraulic heads and time-lapse gravity (TLG) data acroos the domain,
and writes outputs to ascii files in the pyemu working directory. 

"""
import os
import flopy
import pandas as pd
import numpy as np

def coupledmodel_grid(ws, 
           head_time,
           reference_time,
           target_time,
           sep_grid):
    
    """
    Contains Gravi4GW-hybrid and links it to flopy.
    
       Arguments:
       ----------
       ws :  str
          A path to pyemu working directory.
       
       head_time : list of int(s)
                 A list of time step index (or indices) to extract hydraulic heads.
                    
       reference_time : list of int(s)
                      A list of time step index (or indices) to extract reference heads.
                      
       target_time : list of int(s) 
                   A list of time step index (or indices) to extract target heads.   
                   
       sep_grid : str
                Delimiter used to format output(s) as a specific ascii file type.
                This controls how fields are separated in the generated text file (e.g., ',' for CSV).
          
       Returns:
       -------
       ascii_file_path : str
                       Path to ascii file containing hydraulic head and TlG outputs. The output are also stored in a dictionary.
                       
      out_grid : dic
               A dictionary containing the same outputs stored in the ascii file.
      
      *** Note:
            TLG data unit is microGal. The unit of hydraulic heads is sepcified during creating the model
            through flopy.
                        
    """
    
    def ext_inq(sep_grid):
        """
        Returns the appropriate file extension for output files.
        
        Arguments:
        ----------
        sep_grid: str
                The same sep_grid used in `tlg_gw`.
        
        Returns:
        -------
        ext_fnd : str
                The file extension to return output ascii files.      
        
        """
        if sep_grid == ',':
          ext_fnd = '.csv'
        elif sep_grid == '\t':
            ext_fnd = '.tsv'
        elif sep_grid == ' ':
            ext_fnd = '.dat' 
            
        return ext_fnd 
    
    from gravchaw.models.gravi4gw_hybrid import (tlg_hybrid, xy_activeCell, heads_activeCell)
    # load the groundwater model to link flopy and Gravi4GW-hybrid
    sim = flopy.mf6.MFSimulation.load(sim_ws=ws, exe_name=os.path.join(ws, 'mf6'))
    sim.write_simulation() 
    sim.run_simulation() # simulate hydraulic heads
    gwf = sim.get_model()  
    modelgrid_m = gwf.modelgrid # to extract model's spatial grid
    dis=gwf.dis
    sim_unit=dis.length_units.get_data() # to extract model's unit
    hds = gwf.output.head()   # to extract htdraulic heads 
    extracte_times = hds.get_times()
    ext_fnd=ext_inq(sep_grid)
    # extract grids
    out_grid={}
    ## extract head grid
    head_grid = {}
    head_len=len(head_time)
    for h_itime in range(0, head_len):
        # hydraulic heads extracted based on provided time indecies
        ext_head_time=extracte_times[head_time[h_itime]]
        head=hds.get_data(totim=ext_head_time)
        head=head.squeeze()
        head_grid[f'{ext_head_time:03}']=head
    for i, (key_h, array_h) in enumerate(head_grid.items()):
        df_h = pd.DataFrame(array_h)
        head_grid_file = f'head_grid{i}{ext_fnd}'
        head_path = os.path.join(ws, head_grid_file)
        with open(head_path, 'w') as f_h:
            f_h.write(f"{key_h}\n")
            df_h.to_csv(f_h, sep=sep_grid, float_format="%0.10f", na_rep="NaN", index=False, header=False)  
    out_grid={"head_grid":head_grid}                       
    ## calculate and extract TLG grid      
    if reference_time is None and target_time is None:
       return out_grid
    else:
        sto_m = gwf.sto # to extract porosity
        porosity = sto_m.sy._get_data()
        xy_active = xy_activeCell(modelgrid_m, sim_unit)
        x_gravstn = xy_active['xCellcenters']
        y_gravstn = xy_active['yCellcenters']
        z_gravstn = xy_active['top_active']
        delta_g = {}
        delta_g_grid = {}
        tar_len=len(target_time)
        for itime in range(0, tar_len):
            # reference and target heads extracted based on provided time indecies
            ext_refe_time=extracte_times[reference_time[itime]]
            reference_head = hds.get_data(totim=ext_refe_time) 
            ext_targ_time=extracte_times[target_time[itime]]
            target_head = hds.get_data(totim=ext_targ_time)
            model_out = tlg_hybrid(modelgrid_m, reference_head, target_head, 
                                x_gravstn, y_gravstn, z_gravstn, porosity, sim_unit) 
            delta_g[f'{ext_refe_time:03}-{ext_targ_time:03}'] = model_out['gravity']  
            # return TLg as grid, which inactive cells are set to NaN
            nan_array = np.full(np.shape(reference_head), np.nan)
            active_head = heads_activeCell(reference_head, target_head, xy_active, sim_unit)
            nan_array[active_head['indices_tstp']] = delta_g[f'{ext_refe_time:03}-{ext_targ_time:03}']
            nan_array=nan_array.squeeze()
            delta_g_grid[f'{ext_refe_time:03}-{ext_targ_time:03}']=nan_array
        for j, (key_g, array_g) in enumerate(delta_g_grid.items()):
            df_g = pd.DataFrame(array_g)
            grav_grid_file = f'grav_grid{j}{ext_fnd}'
            grav_path = os.path.join(ws, grav_grid_file)
            with open(grav_path, 'w') as f_g:
                f_g.write(f"{key_g}\n")
                df_g.to_csv(f_g, sep=sep_grid, float_format="%0.10f", na_rep="NaN", index=False, header=False)          
        out_grid={"head_grid":head_grid, "grav_grid":delta_g_grid}  
        
        return out_grid          
            