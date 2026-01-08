import os
import pandas as pd
from pathlib import Path
from importlib.resources import files
import flopy

class ModelBridge(object):
    """
    An interface for preparing and managing inputs for the coupled model used in the inversion-framework.

    This class handles all the model input preprocessing required for compatibility with pyemu.
    
    Arguments:
    ----------
    working_direct :  str
        A path to pyemu working directory. It must be the same as `new_d`
        defined in `pyemu.utils.PstFrom(original_d)`.
     
     pstfrom_obj : obj
                 A PEST(++) control file object created using pyemu through `pyemu.utils.PstFrom`.
     
     *** Note: 
             The ModelBridge must be instantiated after the PEST(++) control file object has been created
             via 'pyemu.utils.PstFrom', as it depends on the directory structure defined in the control file
             and is used to attach the core model.
    Example:
    ----------
                pf = pyemu.utils.PstFrom(new_d=template_ws)
                pb = gravchaw.model_bridge.ModelBridge(working_direct=template_ws,
                                        pstfrom_obj=pf)
        
    """
    
    def __init__(self,
                 working_direct,
                 pstfrom_obj):
        self.working_direct = Path(working_direct)
        self.pstfrom_obj = pstfrom_obj
        
    def add_gravCoor(self,
                     grav_obs_filename,
                     coor_filename,
                     index_col=None,
                     x_col=None,
                     y_col=None,
                     z_col=None, 
                     sep_coor=None):
        """
        Defines TLG model output name and prepares gravity sation coordinates for the model.
        
        Arguments:
        ---------- 
       grav_obs_filename : str
                         The name of the file containing TLG observations. The file must be an ascii file.
                         
       *** Note: The first row must be a header, and the first column must contain strings.                 
       
       coor_filename : str
                     The name of the file containing TLG station coordinates. The file must be an ascii file. 
                     
       *** Note: a column of the file must contain strings. This column is used as `index_col`.
                 
       index_col : int
                 A integer to extract the index column. This column is sused as gravity sation names in the model output.
                 Default is `0`.
                      
       x_col : int or str
             An integer or a column label (string) used to extract the station x coordinates.
             Default is `1`.
              
       y_col :int or str
             An integer or a column label (string) used to extract the station y coordinates.
             Default is `2`.
                    
       z_col : int or str
             An integer or a column label (string) used to extract the station z coordinates.
             Default is `3`.
         
       ***Note: TLG station coordinates must be in meter.
       
       sep_coor : Delimiter used to read the station coordinates file (passed to `pd.read_csv`).
                 Deafuslt is `None`.
                 
        Example:
        ---------- 
                    pb = gravchaw.model_bridge.ModelBridge()
                    pb.add_gravCoor(grav_obs_filename='grav.csv',
                                            coor_filename='coord_gravstn.csv', 
                                            index_col=0,
                                            x_col=1, 
                                            y_col=2, 
                                            z_col=3)  
    
        """
        path_to_grav_obs_filename=os.path.join(self.working_direct, grav_obs_filename)
        self.check_exten(path_to_grav_obs_filename)
        self.grav_output_name = grav_obs_filename
        path_to_coor_filename=os.path.join(self.working_direct, coor_filename)
        self.check_exten(path_to_coor_filename)
        if sep_coor==None:
           sep_coor=self.sep_inq(path_to_coor_filename)
        self.grav_coor_df=pd.read_csv(path_to_coor_filename, sep=sep_coor)
        if index_col==None:
            index_col=0
        station_name=self.extrc_coor_cols(index_col, 'index_col')
        self.check_coor_cont(station_name, 'index_col', str, 'string')
        self.station_name=station_name.tolist()
        if x_col==None:
            x_col=1
        x_gravstn=self.extrc_coor_cols(x_col, 'x_col')
        self.check_coor_cont(x_gravstn, 'x_col', (int, float), 'numerical values')
        self.x_gravstn=x_gravstn.tolist()
        if y_col==None:
            y_col=2
        y_gravstn=self.extrc_coor_cols(y_col, 'y_col')
        self.check_coor_cont(y_gravstn, 'y_col', (int, float), 'numerical values')
        self.y_gravstn=y_gravstn.tolist()
        if z_col==None:
            z_col=3
        z_gravstn=self.extrc_coor_cols(z_col, 'z_col')
        self.check_coor_cont(z_gravstn, 'z_col', (int, float), 'numerical values')
        self.z_gravstn=z_gravstn.tolist()

        return {"grav_output_name": self.grav_output_name, "station_name": self.station_name,
                "x_station": self.x_gravstn, "y_station": self.y_gravstn, "z_station": self.z_gravstn}
    
    def add_gravTim(self,
                    reference_time,
                    target_time,
                    pair_type=None):
        """
        Prepares time step indices for the model.
        
        Arguments:
        ---------- 
        reference_time : list of int(s)  
                       A list of time step index (or indices) to extract reference heads. If the reference time is constant
                       (e.g., time step 0 is used to compute Δg), you may pass either a single integer (e.g., [0]) to indicate
                       a constant observation time index or repeat the same integer multiple times, maching the number of 
                       target time indices (e.g., [0, 0, 0]).
                       
        *** Note:
            Time step indices are zero-based, consistent with FloPy convention.
            These values refer to simulation steps used to compute hydraulic heads.
            Users specify the time step indices corresponding to the observation times for Δg. 
            
        target_time : list of int(s)  
                       A list of time step index (or indices) to extract target heads. If the target time is fixed
                       (e.g., time step 0 is used to compute Δg), you may pass either a single integer (e.g., [0]) to indicate
                       a constant observation time index or repeat the same integer multiple times, maching the number of
                       reference time indices (e.g., [0, 0, 0]).
        
       pair_type : str
                 A string to specify whether time indeices, used to model Δg, are fixed or flexible.
                 There are two options, if `fixed_side` is specified, either `reference_time` or `target_time` is constant.
                 If `flexbl_side` is specified, `reference_time` and `target_time` must contain the same number of time indices.
                 Default is `fixed_side`.
        
        
        Returns:
        ----------  
        len_grav_obs : The number of pair heads used to model Δg
        
        reference_time : the list of refrence time indices to be used by the coupled model.
        
        target_time : Pass the list of target time indices to be used by the coupled model.
        
        Examples:
        ---------- 
                     pb = gravchaw.model_bridge.ModelBridge()
                     pb.add_gravTim(reference_time =[2], target_time=[5, 10, 20, 24])
                     OR
                     pb.add_gravTim(reference_time =[2, 2, 2, 2], target_time=[5, 10, 20, 24])
                     
                     pb.add_gravTim(reference_time =[0, 5, 15, 20], target_time=[24])
                     OR
                     pb.add_gravTim(reference_time =[0, 5, 15, 20], target_time=[24, 24, 24, 24])
                     
                     pb.add_gravTim(reference_time =[0, 5, 15, 20], target_time=[5, 10, 20, 24], pair_type='flexbl_side')
                     
         
        """
        
        if pair_type is None:
           pair_type='fixed_side' 
        self.check_time_inputs(reference_time, target_time, pair_type) # check for valid inputs
        ## prepare model time inputs
        ref_len=len(reference_time)
        tar_len=len(target_time)   
        if pair_type=='fixed_side':
            true_pair = (ref_len==tar_len) or (ref_len==1) or (tar_len==1)
            if not true_pair:
                raise ValueError(f'unvalid length for pair_type {pair_type!s}')    
            if  ref_len==1 and tar_len>1:
                self.reference_time=reference_time*tar_len
                self.target_time=target_time
            elif tar_len==1 and ref_len>1:
                 self.target_time=target_time*ref_len 
                 self.reference_time=reference_time
        elif pair_type=='flexbl_side':
             true_pair = ref_len==tar_len
             if not true_pair:
                 raise ValueError(f'unvalid length for pair_type {pair_type!s}')              
        self.len_grav_obs = len(self.reference_time)
        return self.len_grav_obs, self.reference_time, self.target_time
        
    def add_coupmodel_out(self):
        """
        Checks for hydraulic head an one of the primary parametrs of the coupled model and add the coupled model module to
        PEST(++) control file object `pstfrom_obj`.
        """
        # check if hydraulic heads saved 
        sim = flopy.mf6.MFSimulation.load(sim_ws=self.working_direct)
        gwf = sim.get_model()
        oc_ = gwf.oc
        head_data=oc_.head_filerecord._get_data()
        if head_data is None:
            raise ValueError("Head output is not properly configured. Use flopy.mf6.ModflowGwfoc to define"
                             " both head_filerecord and saverecord settings to write and save head data.")
        
        modelout_file= files("gravchaw.models").joinpath("model_out.py")
        
        ws='.'
        self.pstfrom_obj.add_py_function(str(modelout_file),
                            f"tlg_gw(grav_output_name='{self.grav_output_name}', len_grav_obs={self.len_grav_obs}, reference_time={self.reference_time}, target_time={self.target_time}, station_name={self.station_name}, x_gravstn={self.x_gravstn}, y_gravstn={self.y_gravstn}, z_gravstn={self.z_gravstn}, ws='{ws}')") 
    #helper functions 
    def check_exten(self,
                    obj):
        """
        Checks whether the provided file names include an extension.
        
        """
        ext_fnd = Path(obj).suffix.lower()
        if ext_fnd == "":
            raise ValueError('{obj} must have an extension')
        return ext_fnd   
            
    def sep_inq(self,
                obj):
        """
        Specifies the delimiter used to read the ascii file.

        """
        ext = self.check_exten(obj)
        if ext == '.csv':
          sep_fnd = ','
        elif ext == '.tsv':
            sep_fnd = '\t'
        elif ext in ('.txt', '.dat', '.prn', '.log'):
            sep_fnd = r'\s+'  
        else:
            sep_fnd = r'\s+'    
        return sep_fnd 
    
    def extrc_coor_cols(self,
                        obj,
                        obj_name):
        """
        Extracts specific columns from a pandas DataFrame.
        
        """
        if isinstance(obj, int):
            out_coor = self.grav_coor_df.iloc[:, obj]
        elif isinstance(obj, str): 
            out_coor = self.grav_coor_df[obj]
        else:
            raise TypeError(f'{obj_name} must be a cloumn index (int) or label (str)')
        return out_coor
        
    def check_coor_cont(self,
                        obj,
                        obj_name,
                        expected_type,
                        type_name):
        """
        Checks whether the file contents are of the expected type.
        
        """
        if not obj.map(lambda a: isinstance(a, expected_type)).all():
            raise TypeError(f'{obj_name} must contain only {type_name}')
        
    def check_time_inputs(self,
                          reference_time,
                          target_time,
                          pair_type):
        """
        Checks whether the contents of the time list and pair_type are of the expected type.
        """
        if not (self._is_list_of_ints(reference_time)):
           raise TypeError('reference_time must be lists of integers')
        if not (self._is_list_of_ints(target_time)):
           raise TypeError('target_time must be lists of integers')
        if not isinstance(pair_type, str):
            raise TypeError('pair_type must be a string')  
    
    def _is_list_of_ints(self,
                         obj):
        """
        Checks whether the contents of the time list are of the expected type.
        
        """
        return isinstance(obj, list) and all(isinstance(i, int) for i in obj)