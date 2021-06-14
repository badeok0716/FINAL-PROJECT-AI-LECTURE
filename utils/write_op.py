import pickle
import numpy as np
def write_pkl(content, path):
    '''write content on path with path
    Dependency : pickle
    Args:
        content - object to be saved
        path - string
                ends with pkl
    '''
    with open(path, 'wb') as f:
        print("Pickle is written on %s"%path)
        try: pickle.dump(content, f)
        except OverflowError: pickle.dump(content, f, protocol=4)