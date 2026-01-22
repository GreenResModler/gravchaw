"""
This module uses a hybrid approach to calculate TLG data. 

"""

import numpy as np

def conv_unit(sim_unit,
              obj):
    """
    `tlg_hybrid` requires units in meters. This function converts 
    units to meters if necessary.
    
    Arguments:
    ----------
    sim_unit : str
             The unit extracted from created model using flopy.
    
    obj : List 
        A list containing variables that require unit conversion.
    
    Returns:
    -------
    var_to_conv : Variable(s)
                Variable(s) with converted unit.
                 
    """
    unit_conv_factr ={'feet': 0.3048, 'centimeters': 0.01} # flopy units may be "feet" or "centimeters"
    if sim_unit in unit_conv_factr:
        factor=unit_conv_factr[sim_unit]
        for var_to_conv in obj: 
            var_to_conv *=factor  

def xy_activeCell(modelgrid,
                  sim_unit):
    """
    Extracts the model's spatial grid, including the x and y coordinates 
    of the centers and vertices of active cells.
    
    Arguments:
    ----------
    modelgrid : obj
              The object storing model's spatial grid.
    
    sim_unit : str
             The same sim_unit used in `conv_unit`.

    Returns:
    -------
    xy_active : dic
              A dictionary contaning the x, y coordinates of the centers and vertices of active cells,
              aquifer bottom, and active and inactive cell indices.
        
    """
    
    nlayer_ = modelgrid.nlay
    # extract x and y coordintates of cells
    xCellcenters_whole = modelgrid.xyzcellcenters[0]
    xCellcenters_whole = np.repeat(xCellcenters_whole[np.newaxis,:,:], nlayer_, axis=0) 
    xCellvertices_whole = modelgrid.xyzvertices[0]
    xCellvertices_whole = np.repeat(xCellvertices_whole[np.newaxis,:,:], nlayer_, axis=0)
    yCellcenters_whole = modelgrid.xyzcellcenters[1]
    yCellcenters_whole = np.repeat(yCellcenters_whole[np.newaxis,:,:], nlayer_, axis=0)
    yCellvertices_whole = modelgrid.xyzvertices[1]
    yCellvertices_whole = np.repeat(yCellvertices_whole[np.newaxis,:,:], nlayer_, axis=0)
    # extract x and y coordintates of active cells
    idm = modelgrid._idomain
    idm_len = len(idm.shape)
    if idm_len==2:
        idm = np.repeat(idm[np.newaxis,:,:], nlayer_, axis=0)
    indices = np.where(idm>=1) # mdf R1    
    inactive_indices = np.where(idm<=0) # mdf R1
    indices_xvertices= indices[0], indices[1], indices[2]+1       # indices[1]+1 to add a column number to the indices
    indices_yvertices= indices[0], indices[1]+1, indices[2]       # indices[0]+1 to add a row number to the indices
    xCellcenters = xCellcenters_whole[indices]
    xCellvertices_left = xCellvertices_whole[indices]
    xCellvertices_right = xCellvertices_whole[indices_xvertices]
    yCellcenters = yCellcenters_whole[indices]                    
    yCellvertices_left = yCellvertices_whole[indices_yvertices] 
    yCellvertices_right = yCellvertices_whole[indices]
    top_ =  modelgrid.top
    top_ = np.repeat(top_[np.newaxis,:,:], nlayer_, axis=0) 
    top_active = top_[indices]
    botm_ =  modelgrid.botm[0]
    botm_ = np.repeat(botm_[np.newaxis,:,:], nlayer_, axis=0) 
    # convert units to meter if necessary  
    conv_unit(sim_unit, [xCellcenters,
                        xCellvertices_left,
                        xCellvertices_right,
                        yCellcenters,
                        yCellvertices_left,
                        yCellvertices_right,
                        top_active, # add R1
                        botm_])         
    # store the output as dic         
    xy_active = {'xCellcenters': xCellcenters, # x coordinates of center of active cells
                 'xCellvertices_left': xCellvertices_left, # x coordinates of left side of active cells
                 'xCellvertices_right': xCellvertices_right, # x coordinates of right side of active cells
                 'yCellcenters': yCellcenters, # y coordinates of center of active cells
                 'yCellvertices_left': yCellvertices_left, # y coordinates of left side of active cells
                 'yCellvertices_right': yCellvertices_right, # y coordinates of right side of active cells
                 'top_active':top_active, # top of active cells
                 'botm_': botm_, # bottom of cells
                 'indices': indices, # indices of active cells
                 'inactive_indices': inactive_indices # indices of inactive cells
                 }
    
    return xy_active

def heads_activeCell(GW_depth_tstp1,
                     GW_depth_tstp2,
                     xy_active,
                     sim_unit):
    """
    Extracts hydraulic heads of constant active cells active cells, accounts for the influence of
    wet-dry transition cells on TLG calculations, converts hydraulic head units to meters if necessary.
    
    Arguments:
    ----------
    GW_depth_tstp1 : array
              An array of simulated hydraulic heads at reference time.
    
    GW_depth_tstp2 : array
              An array of simulated hydraulic heads at target time.
              
    xy_active : dic
              The output from `xy_activeCell`.
        
    sim_unit : str
              The same sim_unit used in `conv_unit`.
        
    Returns:
    -------
    heads_active : dic
                 A dictionary contaning hydraulic heads of constant active cells and dry-wet transition cells
                 at reference and target times, and updated indices for dry-wet transition cells.
        
    """
    
    inac = 1E+30   # A flag for inactive cells, a flopy convention.
    # handle wetting and drying of cells
    flg_tstp1 = (GW_depth_tstp1==inac)
    flg_tstp2 = (GW_depth_tstp2==inac)
    conv_cells = flg_tstp1 ^ flg_tstp2
    # replace the new inactive cell with the cell's bottom
    GW_depth_tstp1[flg_tstp1 & conv_cells] = xy_active['botm_'][flg_tstp1 & conv_cells]
    GW_depth_tstp2[flg_tstp2 & conv_cells] = xy_active['botm_'][flg_tstp2 & conv_cells]
    # extract hydraulic head of constant active and transition cells
    indices_GW_depth_tstp1 = np.where(GW_depth_tstp1!=inac) 
    GW_depth_tstp1_active = GW_depth_tstp1[indices_GW_depth_tstp1]
    indices_GW_depth_tstp2 = np.where(GW_depth_tstp2!=inac) 
    GW_depth_tstp2_active = GW_depth_tstp2[indices_GW_depth_tstp2] 
    # convert head to meter if necessary
    conv_unit(sim_unit, [GW_depth_tstp1_active,
                        GW_depth_tstp2_active])
    # store the output as dic  
    heads_active = {'indices_tstp': indices_GW_depth_tstp1,  # updated indices for dry-wet transition cells
                    'GW_depth_tstp1_active': GW_depth_tstp1_active, # hydraulic heads of active cells at reference time
                     'GW_depth_tstp2_active': GW_depth_tstp2_active # hydraulic heads of active cells at target time
                     } 
  
    return heads_active

def porosity_activeCell(porosity,
                        heads_active):
    """
    Extracts porosity (specific yield) of cells affecting TLG calculations,
    including constant active cells and dry-wet transition cells.
    
    Arguments:
    ----------
    porosity : array
             An array of the porosity of all cells (specific yield)
    
    heads_active : dic
                 The output from `heads_activeCell`.
                 
    Returns:
    -------
    porosity_active : dic
                    A dictionary contaning porosity of cells affecting TLG calculations. 
                    
    """

    indices_poros = heads_active['indices_tstp']
    porosity_active = porosity[indices_poros]
    porosity_active = {'porosity_active':porosity_active} # porosity of cells affecting TLG calculations
    
    return porosity_active

def domain_obs(xy_active,
               heads_active,
               x_gravstn,
               y_gravstn,
               z_gravstn,
               porosity_active):  
    """
    Converts required inputs for `tlg_hybrid` to 1D arrays
    and creat a seprate dictionary for each input.
    
    Arguments:
    ----------
    xy_active : dic
              The output from `xy_activeCell`.
    
    heads_active : dic
                 The output from `heads_activeCell`.
              
    x_gravstn : list
              The same x_gravstn used in `tlg_gw`.
        
    y_gravstn : list
              The same y_gravstn used in `tlg_gw`.
              
    z_gravstn : list
              The same z_gravstn used in `tlg_gw`.
              
    porosity_active : dic
                    The output from `porosity_activeCell`.            
        
    Returns:
    -------
    head : dic
         A dictionary contaning 1D arrays of hydraulic heads, and x and y coordinates of cells centers.
    
    obs : dic
        A dictionary contaning 1D arrays of x, y, and z coordinates of gravity stations.
                 
    cell : dic
           A dictionary contaning 1D arrays of cell coordinates and porosity.              
    
    """
    # hydraulic head 
    head = {'x_head': xy_active['xCellcenters'].flatten(), 
            'y_head': xy_active['yCellcenters'].flatten(), 
            'GW_depth_tstp1': heads_active['GW_depth_tstp1_active'].flatten(),
            'GW_depth_tstp2': heads_active['GW_depth_tstp2_active'].flatten()}
    # convert station coordinates to numpy arrays
    x_gravstn = np.array(x_gravstn, dtype =np.float64)
    y_gravstn = np.array(y_gravstn, dtype =np.float64)
    z_gravstn = np.array(z_gravstn, dtype =np.float64)
    # observation coordinates
    obs = {'x_obs': x_gravstn.flatten(), 
           'y_obs': y_gravstn.flatten(),
           'z_obs': z_gravstn.flatten()} 
    # cell coordinates as dic
    cell = {'xLeft': xy_active['xCellvertices_left'].flatten(),
            'xRight': xy_active['xCellvertices_right'].flatten(), 
             'yLeft': xy_active['yCellvertices_left'].flatten(),
             'yRight': xy_active['yCellvertices_right'].flatten(),
              'z_Up': head['GW_depth_tstp1'].flatten(),
              'z_Down': head['GW_depth_tstp2'].flatten(), 
              'poros': porosity_active['porosity_active'].flatten()} 
          
    return head, obs, cell

def g_point_mass(R_cen,
                 delX,
                 delY,
                 delZ,
                 dx_cen,
                 dy_cen,
                 dz_cen,
                 cell_pors):
    """
    Calculates TLG using point mass method for cells far from gravity stations.
    
    Arguments:
    ----------
    R_cen : array
          1D array of the distance between gravity stations and cell centers.
    
    delX : array
         1D array of x dimension of far cells.
              
    delY : array
         1D array of y dimension of far cells.
        
    delZ : array
         1D array of z dimension of far cells.
              
    dx_cen : array
           1D array of the x component of the distance between gravity stations and cell center.
              
    dy_cen : array
           1D array of the y component of the distance between gravity stations and cell center.
           
    dz_cen : array
           1D array of the z component of the distance between gravity stations and cell center.
    
    cell_pors : array
              1D array of porosity of cells.
  
    Returns:
    -------
    dgpoZ : array
          1D array of calculated TLG (in microGal) for far cells.
   
    """
    dv = delX*delY*delZ
    dgp2z = dz_cen/(R_cen**3)
    dgpoZ = dv*dgp2z*cell_pors
    dgpoZ = -dgpoZ
    
    return dgpoZ

def g_cell(x_diff,
           y_diff,
           z_diff,
           cell_pors): 
    """
    Calculates TLG using rectangular prsim method.
    
    Arguments:
    ----------
    x_diff : array
          2D array of the distance between x coordiantes of gravity stations and cell vertical faces.
    
    y_diff : array
          2D array of the distance between y coordiantes of gravity stations and cell vertical faces.
              
    z_diff : array
          2D array of the distance between z coordiantes of gravity stations and cell horizontal faces.
    
    cell_pors : array
              1D array of porosity of cells.
  
    Returns:
    -------
    dgprZ : array
          1D array of calculated TLG (in microGal) for close cells.
    
    """
    sum = 0
    for i in np.arange(1, 3):
        for j in np.arange(1, 3):
            for k in np.arange(1, 3):
                mu = (-1)**(i+j+k)
                dx = x_diff[:,i-1]
                dy = y_diff[:,j-1]
                dz = z_diff[:,k-1] 
                dz[dz>-1E-06] = -1E-06  # to avoid singularity
                R = np.sqrt(dx**2+dy**2+dz**2)
                t = (dx*dy)/(dz*R)
                L1 = R+dx
                L2 = R+dy
                x_term = dx*np.log(L2)
                y_term = dy*np.log(L1)
                z_term = dz*np.arctan(t)
                sum = sum + mu*(z_term-x_term-y_term)*cell_pors
                dgprZ = -sum
                
    return dgprZ  
           
def tlg_hybrid(modelgrid,
                    GW_depth_tstp1,
                    GW_depth_tstp2,
                    x_gravstn,
                    y_gravstn,
                    z_gravstn,
                    porosity,
                    sim_unit):
    """
    Calculates TLG data using rectangular prism and poit mass methods.
    
    Arguments:
    ----------
    modelgrid : obj
              The same modelgrid used in `xy_activeCell`.
    
    GW_depth_tstp1 : array
                   The same GW_depth_tstp1 used in `heads_activeCell`.
              
    GW_depth_tstp2 : array
                   The same GW_depth_tstp2 used in `heads_activeCell`.
    
    x_gravstn : list of float or int
              A list of x coordinates (in meter) of gravity stations.
              
    y_gravstn : list of float(s) or int(s)
              A list of y coordinates (in meter) of gravity stations.   
    
    z_gravstn : list of float or int
              A list of z coordinates (in meter) of gravity stations. 
                               
    porosity : array
             The same porosity used in `porosity_activeCell`.
        
    sim_unit : str
             The same sim_unit used in `conv_unit`.
  
    Returns:
    -------
    model_out : dic
              A dictionary containting calculated TLG (in microGal).
    
    
    """
    G = 6.67408e-11  # m3/kg.s2                       
    rho_H2O = 1000   # kg/m3
    # call active cell x, y coordinates
    xy_active = xy_activeCell(modelgrid, sim_unit)
    # call active cell heads
    heads_active = heads_activeCell(GW_depth_tstp1, GW_depth_tstp2, xy_active, sim_unit)
    # call porosity of active heads
    porosity_active = porosity_activeCell(porosity, heads_active)
    # call structure arrays
    head, obs, cell = domain_obs(xy_active, heads_active, x_gravstn, y_gravstn, z_gravstn, porosity_active)
    delX = cell['xRight']-cell['xLeft']
    delY = cell['yRight']-cell['yLeft']
    delZ = head['GW_depth_tstp2']-head['GW_depth_tstp1']
    Del_P2= delX**2 + delY**2 + delZ**2  
    z_cen = (head['GW_depth_tstp1']+head['GW_depth_tstp2'])/2
    nstn = np.size(obs['x_obs'])
    gz = np.zeros(nstn)
    for n in np.arange(nstn):
        x_diff = np.array([cell['xLeft']-obs['x_obs'][n], cell['xRight']-obs['x_obs'][n]])
        x_diff = x_diff.transpose()
        y_diff = np.array([cell['yLeft']-obs['y_obs'][n], cell['yRight']-obs['y_obs'][n]])
        y_diff = y_diff.transpose()
        z_diff = np.array([cell['z_Up']-obs['z_obs'][n], cell['z_Down']-obs['z_obs'][n]])
        z_diff = z_diff.transpose()
        dx_cen = head['x_head'] - obs['x_obs'][n]
        dy_cen = head['y_head'] - obs['y_obs'][n]
        dz_cen = z_cen - obs['z_obs'][n]    
        dz_cen[dz_cen>-1E-06] = -1E-06 # to avoid singularity
        R_cen = np.sqrt(dx_cen**2+dy_cen**2+dz_cen**2)
        f2 = (R_cen**2)/Del_P2
        # Decide the method  
        points_ind = np.where(f2>81)[0]
        dg_point = g_point_mass(R_cen[points_ind],
                                delX[points_ind], 
                                delY[points_ind],
                                delZ[points_ind],
                                dx_cen[points_ind], 
                                dy_cen[points_ind],
                                dz_cen[points_ind], 
                          cell['poros'][points_ind])
        cell_ind = np.where(~(f2>81))[0]
        dg_cell = g_cell(x_diff[cell_ind],
                         y_diff[cell_ind],
                         z_diff[cell_ind],
                         cell['poros'][cell_ind]) 
        sum_g = np.sum(dg_point) + np.sum(dg_cell)      
        dgz = sum_g*G*rho_H2O
        dgz_uGal = dgz*1E08  # convert to uGal
        gz[n]= dgz_uGal
        model_out = {'gravity':gz} # calculate TLG for all model cells
    
    return model_out
