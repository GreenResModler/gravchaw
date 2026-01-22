"""
This module executes the coupled hydrogravimetric model at the post optimization stage.

"""
def tlg_gw_g(ws, 
           head_time,
           reference_time,
           target_time,
           sep_grid):
    
    from gravchaw.models.coupled_model_grid import coupledmodel_grid
    coupledmodel_grid(ws, 
                head_time,
                reference_time,
                target_time,
                sep_grid)
  