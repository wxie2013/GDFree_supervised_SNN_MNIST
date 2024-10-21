## this module should be identical for all copies of Brian2 packages. 
from pathlib import Path
import importlib, sys

def get_brian_head():
    try:
        brian_dir = str(Path.cwd())
        brian_dir = brian_dir[:brian_dir.index('Brian2')+len('Brian2')]
        path_head = str(Path.cwd().relative_to(Path.home()))
        path_head = path_head[:path_head.index('Brian2')+len('Brian2')].replace('/', '.')
    except:
        brian_dir = '/home/wxie/Brian2'
        path_head = 'Brian2'

    return brian_dir, str(path_head)

#
brian_dir, path_head = get_brian_head()
#print('--- brian_dir: ', brian_dir, ',  path_head: ', path_head, ' ---')
