# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:20:48 2021
@author: Joseph Vermeil

MainDepthoMaker_##.py - Script to use the Deptho-making functions in the BeadTracker program.
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

# %% General imports

# 1. Imports
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

import os
import sys
import matplotlib

# Local Imports

import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun

from BeadTracker import depthoMaker

# 2. Pandas settings
pd.set_option('mode.chained_assignment',None)

# 3. Graphical settings
gs.set_default_options_jv()


# %% Next depthos !

# %% Next depthos !

# %% Example Deptho making - All depthos from 22.07.27 Long Linker experiments

DirDataRaw = 'D://MagneticPincherData//Raw'
DirDataRaw = cp.DirDataRaw
date = '22.07.27'



#### M1
subdir = 'M1'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M1_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


#### M2
subdir = 'M2'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M2_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


#### M3
subdir = 'M3'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M3_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)


#### M4
subdir = 'M4'
DirDataRawDate_Deptho = os.path.join(DirDataRaw, date + '_Deptho', subdir)
DirDataRawDepthoLibrary = os.path.join(DirDataRaw, 'DepthoLibrary')

specif = 'all' # can be 'all' or any string that you want to have in the deptho file name
beadType = 'M450'
saveLabel = date + '_M4_M450_step20_100X'
# convention - saveLabel = 'date_manip_beadType_stepSize_otherSpecs'
scale = gc.SCALE_100X # pix/µm

depthoMaker(DirDataRawDate_Deptho, DirDataRawDepthoLibrary, 
            specif, saveLabel, scale, beadType = beadType, step = 20, d = 'HD', plot = 0)

