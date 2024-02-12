from dataclasses import dataclass
import numpy as np

@dataclass
class parameters:
  def __init__(self):
    self.sigma = 0.1
    self.params = {
    "r_x": np.float32(1.0),
    "r_y": np.float32(1.0),
    "K_x": np.float32(1.0),
    "K_y": np.float32(1.0),
    "beta": np.float32(0.3),
    "v0":  np.float32(0.3),
    "D": np.float32(1.1),
    "tau_yx": np.float32(0.0),
    "tau_xy": np.float32(0.0),
    "cV": np.float32(0.1),
    "f": np.float32(0.1),
    "dH": np.float32(0.1),
    "alpha": np.float32(1),
    "sigma_x": np.float32(self.sigma),
    "sigma_y": np.float32(self.sigma),
    "sigma_z": np.float32(self.sigma)
    }
    self.reset_params = self.params
  
  def parameters(self):
    return self.params
  
  def reset(self):
    self.params = self.reset_params
    return self.params
  
  def init_state(self):
    return np.array([0.5,0.5,0.7], dtype=np.float32)

  
 
    
