"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import numpy as np

def scale_data(data):
  data = np.array(data)
  print(isinstance(data,float))
  data_mask = data[(data>=0) & (data<=1) & isinstance(data,float)]
  print(data_mask)
  return data_mask.all()