# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:31:26 2022
@author: Joseph Vermeil

MainAnalyzer_##.py - Script to use the TrackAnalyzer program.
Please replace the "_NewUser" in the name of the file by "_##", 
a suffix corresponding to the user's name (ex: JV for Joseph Vermeil) 
Joseph Vermeil, 2022

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

# %% > Imports and constants

#### Main imports

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt


import os
import sys
import matplotlib


#### Local Imports

import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)


import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka

#### Potentially useful lines of code
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# cp.DirDataFigToday

#### Pandas
# pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')


####  Matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})

#### Graphic options
gs.set_default_options_jv()



# %% TimeSeries functions

# %%% List files
allTimeseriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                          if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
    
print(allTimeseriesDataFiles)

# %%% Get a time series
# Example : df = taka.getCellTimeSeriesData('22-02-09_M1_P1_C7')


# %%% Plot a time series
#  Example : taka.plotCellTimeSeriesData('21-02-10_M1_P1_C2')




# #############################################################################
# %% GlobalTables functions

# =============================================================================
# %%% Experimental conditions
expDf = ufun.getExperimentalConditions(cp.DirRepoExp, save=True , sep = ';')



# =============================================================================
# %%% Constant Field

# %%%% Update the table
# Example : taka.computeGlobalTable_ctField(task='updateExisting', fileName = 'Global_CtFieldData_Py', 
#                                           save = False, source = 'Python')

# %%%% Refresh the whole table
# Example : taka.computeGlobalTable_ctField(task = 'fromScratch', fileName = 'Global_CtFieldData_Py', 
#                                           save = True, source = 'Python')

# %%%% Display
# Example : df = taka.getGlobalTable_ctField().head()



# =============================================================================
# %%% Mechanics

# %%%% Update the table

# Example : 
# taka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_Py', 
#                             save = False, PLOT = False, source = 'Matlab') # task = 'updateExisting'


# %%%% Refresh the whole table

# Example : 
# taka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_Py2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% Specific task

# Example : 
# Task_1 = '22-05-03'
# taka.computeGlobalTable_meca(task = Task_1, fileName = 'Global_MecaData_Demo', 
#                             save = True, PLOT = True, source = 'Python') # task = 'updateExisting'
# Task_2 = '22-05-03_M1 & 22-05-04_M2'
# taka.computeGlobalTable_meca(task = Task_2, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'
# Task_3 = '22-05-03 & 22-05-04 & 22-05-05'
# taka.computeGlobalTable_meca(task = Task_3, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'



# %%%% Display

# Example : df = taka.getGlobalTable_meca('Global_MecaData_Py2').tail()



# =============================================================================
# %%% Fluorescence

# %%%% Display

# Example : df = taka.getFluoData().head()


