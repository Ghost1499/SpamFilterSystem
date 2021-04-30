import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SystemController(object):
    """description of class"""
    def __init__(self):
        pass

    def train_from_file(self,filename):
        df=pd.read_csv(filename)
        # df.drop('Unnamed: 0', axis=1, inplace = True)
        t = np.arange(0, 2.5, 0.5)
        y1 = np.sin(math.pi * t)
        y2 = np.sin(math.pi * t + math.pi / 2)
        y3 = np.sin(math.pi * t - math.pi / 2)
        print(t)
        for value in t:
            print(value,y3[value])
            plt.text(value,y3[value],y3[value])
        plt.plot(t, y1, 'b--', t, y2, 'g', t, y3, 'r-.')
        plt.show()
        #print(df)

