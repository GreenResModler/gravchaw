"""
This module executes the coupled hydrogravimetric model in the optimization process.

"""
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