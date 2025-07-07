import os 
from natsort import natsorted

def list_filetype_alphanumeric_order(directory, filetype):
    """
    List files in a directory with a specific filetype, sorted alphanumerically.
    """

    files = [f for f in os.listdir(directory) if f.endswith(filetype)]
    return natsorted(files)