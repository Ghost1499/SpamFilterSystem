import os
from typing import List


def make_path(path:List[str],filename:str=""):
    path="/".join(path)
    if filename and not path.endswith(('/','\\','//','\\\\')):
        path+='/'
    return path+filename
