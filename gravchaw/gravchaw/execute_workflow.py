"""
This module run the optimization/uncertainty analysis algorithm and implements post optimization by
updating parametrs in the pyemu working directory.
"""
import pyemu
import subprocess
from pathlib import Path
from . models.model_grid import tlg_gw_g

def write_script(file_name,
                 working_direct,
                 pst_file_name):
    """
    Writes a script for updating parameter files in the working directory.
    The generated script can be executed to apply updates to parameter files.
    Functions implementing these updates are defined in pyemu notebooks.
    
    Arguments:
    ----------
    file_name : str
              The name of the generated script. This is a fixed name and is not exposed to the user interface.
    
    working_direct : str
                   A path to pyemu working directory.
                 
    pst_file_name : str
                  The name of pst file used to run the algorithm.
                   
    Returns:
    -------
    file_path : str
              Path to the generated Python script (ending with a `.py` extension).
 
    """

    target_dir = Path(working_direct).resolve() # The script is returnd to the working directory
    filepath = target_dir / file_name

    # write the script
    script = f"""\
import os
from pathlib import Path
import pyemu

def script_to_opt_post_opt(working_direct, pst_file):

    pst_path = os.path.join(working_direct, f"{{pst_file}}.pst")
    pst = pyemu.Pst(pst_path)

    par_file = os.path.join(working_direct, f"{{pst_file}}.par")
    pst.parrep(par_file)
    pst.write_input_files(pst_path=working_direct)
    pyemu.os_utils.run('python forward_run.py', cwd=working_direct)

def main():
    # Inputs are passed here
    working_direct = {repr(working_direct)}
    pst_file = {repr(pst_file_name)}
    script_to_opt_post_opt(working_direct, pst_file)

if __name__ == "__main__":
    main()
"""
    filepath.write_text(script)
    print(f"Script written to: {filepath}")
    
def run(cmd_str,
        ws,
        verbose=None,
        use_sp = False,
        post_updtpar=False, # Note: valid only for inversion algorithms (currently GLM; IES not yet validated).
        head_time=None,
        reference_time=None,
        target_time=None,
        sep_grid=None,
        **kwargs):
    """
    Runs the specified algorithm.

    If post-optimization processing is enabled, the function also updates the 
    model parameters and calculated hydraulic head and TLG across the domain.
    This process is valid only for inversion algorithms.
    
       Arguments:
       ----------
       cmd_str : str
               The algorithm to execute (e.g. f'pestpp-glm file_name.pst').
          
       ws : str
          A path to pyemu working directory. Default is '.'.
          
       verbose : bool
                Controls whether command-line execution details (`cmd_str`) are printed to stdout.  
                This option is handled internally by 'os_utils.run', a utility from the pyemu package.  
                Default is `False`.
                
       use_sp: bool
             Controls how the external system command is executed.
             This option passed directly to `pyemu.os_utils.run`, where it determines whether
             `os.system()` or `subprocess.Popen` is used.
               
       post_updtpar : bool
                    Flag to enable post-optimization process to update input paramters with optimal parameters.
                    Default is `False`.
                            
       
       head_time : list of int(s)
                 A list of time step index (or indices) to extract hydraulic heads.
                 Only required if `post_updtpar` is `True`.
                    
       reference_time : list of int(s)
                      A list of time step index (or indices) to extract reference heads.
                      Only required if `post_updtpar` is `True`, and `target_time` is provided.
                      
       target_time : list of int(s) 
                   A list of time step index (or indices) to extract target heads. 
                   Only required if `post_updtpar` is `True`, and `reference_time` is provided.
                   
       sep_grid : str
                Delimiter used to format output(s) as a specific ascii file type.
                This controls how fields are separated in the generated text file (e.g., ',' for CSV).
                Only required if `post_updtpar` is `True`. Deafult is ','.
          
       Returns:
       -------
       Specific returns if post_updtpar is enabled:
           
       ascii_file_path : str
                       Path to ascii file containing hydraulic heads and TlG outputs across the domain.
                              
       out_grid : dic
               A dictionary containing the same outputs stored in the ascii file.           
    
    """
    if verbose==None:
        verbose=False
    if ws is None:
        ws='.'
    if post_updtpar==True:
        # run the algorithm
        if head_time==None:
            raise ValueError("head_time is required")
        if (reference_time is None) != (target_time is None):
            raise ValueError("Both 'reference_time' and 'target_time' must be provided or both omitted."
                             "If omitted, only the hydraulic head grid will be returned.")
        if reference_time is not None and len(reference_time) != len(target_time):
            raise ValueError("To return TLG grid, 'reference_time' and 'target_time' must have the same length.")
        if sep_grid==None:
            sep_grid=','
        # run the algorithm and perform post-optimization process    
        pyemu.os_utils.run(cmd_str=cmd_str,
                           cwd=ws,
                           verbose=verbose,
                           use_sp = False,
                           **kwargs)
        # extract pst name
        pst_name=cmd_str.split()[1].replace(".pst", "")
        write_script("post_script.py", ws, pst_name)
        #run post-optimization
        subprocess.run(["python", f'{ws}/post_script.py']) 
        out_grid =tlg_gw_g(ws=ws,
                         head_time=head_time,
                         reference_time=reference_time,
                         target_time=target_time,
                         sep_grid=sep_grid)
        return out_grid

    else:
        pyemu.os_utils.run(cmd_str=cmd_str,
                           cwd=ws,
                           verbose=verbose,
                           use_sp = False,
                           **kwargs)