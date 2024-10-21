## this module should be identical for all copies of Brian2 packages. 
from Brian2.get_brian_dir_path_head import *

common_module = importlib.import_module(str(path_head)+'.common_module')
names = [x for x in common_module.__dict__ if not x.startswith("_")]
globals().update({k: getattr(common_module, k) for k in names})
