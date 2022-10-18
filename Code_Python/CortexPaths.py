# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:51:13 2022
@authors: Joseph Vermeil, Anumita Jawahar

CortexPaths.py - state all the paths to folders and files used by CortExplore programs, 
to be imported with "import CortexPaths as cp" and call the constants with "cp".my_path".
Joseph Vermeil, Anumita Jawahar, 2022

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

# %% 0. Imports

import os
import sys
import ctypes
from datetime import date

COMPUTERNAME = os.environ['COMPUTERNAME']
print(COMPUTERNAME)

# %% 1. Paths

# 1.1 Init main directories

if COMPUTERNAME == '>>>Computer_name':
    suffix = '>>>_##'
    DirRepo = "C://Users//>>>User//Desktop//CortExplore"
    DirData = "D://MagneticPincherData"
    DirCloud = "C://Users//>>>User//>>>ownCloud//MagneticPincherData" + suffix
    DirTempPlots = "C://Users//>>>User//Desktop//TempPlots"
    CloudSaving = '>>>OwnCloud'
    
    
elif COMPUTERNAME == '>>>Other_computer_name':
    suffix = '>>>_##'
    DirRepo = "C://Users//>>>User//Desktop//CortExplore"
    DirData = "D://MagneticPincherData"
    DirCloud = "C://Users//>>>User//>>>ownCloud//MagneticPincherData" + suffix
    DirTempPlots = "C://Users//>>>User//Desktop//TempPlots"
    CloudSaving = '>>>OwnCloud'
    
    
# 1.2 Init sub directories

DirRepoPython = os.path.join(DirRepo, "Code_Python")
DirRepoPythonUser = os.path.join(DirRepoPython, "Code" + suffix)
DirRepoExp = os.path.join(DirRepo, "Data_Experimental" + suffix)

DirDataRaw = os.path.join(DirData, "Raw")
DirDataRawDeptho = os.path.join(DirDataRaw, 'DepthoLibrary')
DirDataRawDepthoInter = os.path.join(DirDataRawDeptho, 'IntermediateSteps')

# DirDataExp = os.path.join(DirData, "Data_Experimental")

DirDataAnalysis = os.path.join(DirData, "Data_Analysis")
DirDataAnalysisUMS = os.path.join(DirDataAnalysis, "UserManualSelection")
DirDataTimeseries = os.path.join(DirData, "Data_Timeseries")
DirDataTimeseriesRawtraj = os.path.join(DirDataTimeseries, "Trajectories_raw")
DirDataTimeseriesTraj = os.path.join(DirDataTimeseries, "Trajectories")
DirDataTimeseriesStressStrain = os.path.join(DirDataTimeseries, "Timeseries_stress-strain")

DirDataFig = os.path.join(DirData, "Figures")
DirDataFigToday = os.path.join(DirDataFig, "Historique", str(date.today()))

if not CloudSaving == '':
    DirCloudExp = os.path.join(DirCloud, "Data_Experimental")
    DirCloudAnalysis = os.path.join(DirCloud, "Data_Analysis")
    DirCloudAnalysisUMS = os.path.join(DirCloudAnalysis, "UserManualSelection")
    DirCloudTimeseries = os.path.join(DirCloud, "Data_Timeseries")
    DirCloudTimeseriesStressStrain = os.path.join(DirCloudTimeseries, "Timeseries_stress-strain")
    DirCloudFig = os.path.join(DirCloud, "Figures")
    DirCloudFigToday = os.path.join(DirCloudFig, "Historique", str(date.today()))
else:
    DirCloudExp, DirCloudAnalysis, DirCloudAnalysisUMS = "", "", "" 
    DirCloudTimeseries, DirCloudTimeseriesStressStrain = "", ""
    DirCloudFig, DirCloudFigToday = "", ""

# 1.3 Add python directory to path

sys.path.append(DirRepoPython)



# %% 2. Useful functions

MainDirs = [DirRepo, DirData, DirTempPlots]
RepoSubdirs = [DirRepoPython, DirRepoPythonUser, DirRepoExp]
DataSubdirs = [DirDataRaw, DirDataAnalysis, DirDataFig,
               DirDataTimeseries, DirDataTimeseriesTraj, DirDataTimeseriesRawtraj, DirDataTimeseriesStressStrain]

if not CloudSaving == '':
    CloudDirs = [DirCloud, DirCloudExp, DirCloudFig, DirCloudAnalysis, DirCloudTimeseries, DirCloudTimeseriesStressStrain]


def checkDirArchi():
    """
    Check if the local file architecture is well defined and existing.
    """
    valid_main = True
    for p in MainDirs:
        if not os.path.exists(p):
            print(p)
            valid_main = False
    
    if not valid_main:
        print('One of the main directories is missing')
        
    else:
        valid_repo, valid_data, valid_cloud = True, True, True
        
        for p in RepoSubdirs:
            if not os.path.exists(p):
                print(p)
                valid_repo = False
        if not valid_repo:
            print('One of the repository sub-directories is missing')
                
        for p in DataSubdirs:
            if not os.path.exists(p):
                print(p)
                valid_repo = False
        if not valid_repo:
            print('One of the data sub-directories is missing')
        
        if not CloudSaving == '':
            for p in CloudDirs:
                if not os.path.exists(p):
                    print(p)
                    valid_cloud = False
        if not valid_cloud:
            print('One of the cloud sub-directories is missing')
            
    if valid_main and valid_repo and valid_data and valid_cloud:
        print('Directories architecture is correct !')
            
            
            
            
def makeDirArchi():
    """
    Create all the folders missing to the file architecture on this computer.
    """
    for p in MainDirs:
        if not os.path.exists(p):
            os.makedirs(p)
    
    for p in RepoSubdirs:
        if not os.path.exists(p):
            os.makedirs(p)
            
    for p in DataSubdirs:
        if not os.path.exists(p):
            os.makedirs(p)

    if not CloudSaving == '':
        for p in CloudDirs:
            if not os.path.exists(p):
                os.makedirs(p)
        
        warningCloudExp = os.path.join(DirCloudExp, 'Warning.txt')
        if not os.path.exists(warningCloudExp):
            f = open(warningCloudExp, "w")
            text = 'WARNING\nDo not modify this file. It is for consultation only.\n'
            text += 'For the modifiable version go to: ' + DirRepoExp
            f.write(text)
            f.close()

                
                
# %% Final Architecture - august 2022

# Notes:
# > ## is the user's suffix (for example: JV: Joseph Vermeil).

# C:/
# ├─ Users/
# │  ├─ User/
# │  │  ├─ Desktop/
# │  │  │  ├─ CortExplore/
# │  │  │  │  ├─ Data_BeadsCalibration/
# │  │  │  │  │  ├─ Contains files regarding the calibration of beads lots.
# │  │  │  │  │ 
# │  │  │  │  ├─ Data_Experimental/
# │  │  │  │  │  ├─ The VERY IMPORTANT experimental data table (.csv).
# │  │  │  │  │ 
# │  │  │  │  ├─ Code_Matlab/
# │  │  │  │  │  ├─ Any matlab coded program. For now mainly the Photomask drawing.
# │  │  │  │  │ 
# │  │  │  │  ├─ Code_Python/
# │  │  │  │  │  ├─ Scripts_##/ 		 -> TO CREATE AT THE BEGINNING FROM "Scripts_NewUser". Personnal code! Code to analyse, or to plot the data. Mainly scripts.
# │  │  │  │  │  ├─ Scripts_NewUser/	 -> Similar to Scripts_## but contains ~empty scripts, ready to be copied and used in case there is a new user for this code.
# │  │  │  │  │  ├─ ImagesPreprocessing.py -> Raw images on external drives TO croped images on local drive.
# │  │  │  │  │  ├─ BeadTracker.py	 -> Croped images on local drive TO timeseries (.csv) files (see below).
# │  │  │  │  │  ├─ TrackAnalyser.py	 -> Timeseries (.csv) files TO complex mechanical analysis, results saved in large tables (.csv).
# │  │  │  │  │  ├─ UtilityFunctions.py	 -> Many subfunctions called by diverse programms.
# │  │  │  │  │  ├─ CortexPaths.py	 -> Sets all the paths depending of the computer being used to run the code.
# │  │  │  │  │  ├─ GlobalConstants.py	 -> Contains all the important constants common to all programms.
# │  │  │  │  │  ├─ GraphicStyles.py	 -> Contains useful variables and functions for plotting data in other programms.
# │  │  │  │  │  ├─ BeadsCalibration.py	 -> Contains the function used to calibrate a new lot of beads.
# │  │  │  │  │ 
# │  │  │  │  ├─ LICENSE
# │  │  │  │  ├─ README.md
# │  │  │  │  ├─ .git/
# │  │  │  │ 
# │  │  │  │ 
# │
# D:/
# ├─ MagneticPincherData/
# │  ├─ Data_Analysis/
# │  │  ├─ All the large analysed data tables (.csv).
# │  │ 
# │  ├─ Data_Experimental/
# │  │  ├─ The VERY IMPORTANT experimental data table (.csv).
# │  │ 
# │  ├─ Data_Timeseries/
# │  │  ├─ All timeseries files (.csv) -> For each cell: F, B, dx, dy, dz, D3 as function of T.
# │  │  ├─ Trajectories_raw/
# │  │  │  ├─ All trajectories_raw files (.csv) -> For each bead: x, y, z as function of T, in an intermediate step of the tracking.
# │  │  ├─ Trajectories/
# │  │  │  ├─ All trajectories files (.csv) -> For each bead: x, y, z as function of T, in the end of the tracking.
# │  │  ├─ Data_Timeseries_stress-strain/
# │  │  │  ├─ All timeseries_stress-strain files (.csv) -> Same as timeseries BUT also with H0, stress & strain!
# │  │ 
# │  ├─ Figures/
# │  │  ├─ Historique/
# │  │  │  ├─ Folders for each dates with the figures of this days
# │  │  ├─ Folders for various themes/projects...
# │  │  │  ├─ Figures related to this theme/project...
# │  │ 
# │  ├─ Raw/
# │  │  ├─ DepthoLibrary/
# │  │  │  ├─ Intermediate_Py/
# │  │  │  ├─ All the final depthographs images (.tif).
# │  │  ├─ yy.mm.dd/	  -> Raw timelapse images (.tif), Fields.txt and .Results.txt files for this experiment day.
# │  │  ├─ yy.mm.dd_Deptho/ -> Raw deptho stacks (.tif) and .Results.txt files for this experiment day.
# │  │  ├─ yy.mm.dd_Fluo/   -> Potentially, fluo images (.tif) as extracted from the timelapse.
