# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:07:45 2022
@author: Joseph Vermeil & Anumita Jawahar

UtilityFunctions.py - 
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

# %% (0) Imports and settings

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

import re
import os
import sys
import time
import random
import itertools

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import UtilityFunctions as ufun

# %%% Smaller settings

# Pandas settings
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

# Plot settings
gs.set_default_options_jv()


####

dictSubstrates = {}
for i in range(5,105,5):
    dictSubstrates['disc' + str(i) + 'um'] = str(i) + 'um fibronectin discs'
    dictSubstrates['disc{:02.0f}um'.format(i)] = str(i) + 'um fibronectin discs'

                    
# %% (1) TimeSeries functions

def getCellTimeSeriesData(cellID, fromPython = True):
    if fromPython:
        allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) 
                              if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) 
                                  and f.endswith("PY.csv"))]
    else:
        allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) 
                              if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) 
                                  and f.endswith(".csv") and not f.endswith("PY.csv"))]
    fileFound = False
    nFile = len(allTimeSeriesDataFiles)
    iFile = 0
    while (not fileFound) and (iFile < nFile):
        f = allTimeSeriesDataFiles[iFile]
        if f.startswith(cellID + '_'):
            timeSeriesDataFilePath = os.path.join(cp.DirDataTimeseries, f)
            timeSeriesDataFrame = pd.read_csv(timeSeriesDataFilePath, sep=';')
            fileFound = True
        iFile += 1
    if not fileFound:
        timeSeriesDataFrame = pd.DataFrame([])
    else:
        for c in timeSeriesDataFrame.columns:
                if 'Unnamed' in c:
                    timeSeriesDataFrame = timeSeriesDataFrame.drop([c], axis=1)
    return(timeSeriesDataFrame)

def plotCellTimeSeriesData(cellID, fromPython = True):
    X = 'T'
    Y = np.array(['B', 'F', 'dx', 'dy', 'dz', 'D2', 'D3'])
    units = np.array([' (mT)', ' (pN)', ' (µm)', ' (µm)', ' (µm)', ' (µm)', ' (µm)'])
    timeSeriesDataFrame = getCellTimeSeriesData(cellID, fromPython)
    print(timeSeriesDataFrame.shape)
    # my_default_color_cycle = cycler(color=my_default_color_list)
    # plt.rcParams['axes.prop_cycle'] = my_default_color_cycle
    if not timeSeriesDataFrame.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
        axes = timeSeriesDataFrame.plot(x=X, y=Y, kind='line', ax=None, subplots=True, sharex=True, sharey=False, layout=None, \
                       figsize=(8,10), use_index=True, title = cellID + ' - f(t)', grid=None, legend=False, style=None, logx=False, logy=False, \
                       loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, \
                       table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False)
        plt.gcf().tight_layout()
        for i in range(len(Y)):
            axes[i].set_ylabel(Y[i] + units[i])
        # plt.gcf().show()
        plt.show()
    else:
        print('cell not found')
    # plt.rcParams['axes.prop_cycle'] = my_default_color_cycle
        
def getCellTrajData(cellID, Ntraj = 2):
    trajDir = os.path.join(cp.DirDataTimeseries, 'Trajectories')
    allTrajFiles = [f for f in os.listdir(trajDir) 
                    if (os.path.isfile(os.path.join(trajDir, f)) 
                        and f.endswith(".csv"))]
    fileFound = 0
    nFile = len(allTrajFiles)
    iFile = 0
    listTraj = []
    while (fileFound < Ntraj) and (iFile < nFile):
        f = allTrajFiles[iFile]
        if f.startswith(cellID):
            fileFound += 1
            trajFilePath = os.path.join(trajDir, f)
            trajDataFrame = pd.read_csv(trajFilePath, sep='\t')
            for c in trajDataFrame.columns:
                if 'Unnamed' in c:
                    trajDataFrame = trajDataFrame.drop([c], axis=1)
            
            pos = 'na'
            if 'In' in f:
                pos = 'in'
            elif 'Out' in f:
                pos = 'out'
            
            dictTraj = {}
            dictTraj['df'] = trajDataFrame
            dictTraj['pos'] = pos
            dictTraj['path'] = trajFilePath
            
            listTraj.append(dictTraj)
            
        iFile += 1

    return(listTraj)
        
def addExcludedCell(cellID, motive):
    f = open(os.path.join(cp.DirRepoExp, 'ExcludedCells.txt'), 'r')
    lines = f.readlines()
    nLines = len(lines)
    excludedCellsList = []
    for iLine in range(nLines):
        line = lines[iLine]
        splitLine = line[:-1].split(',')
        excludedCellsList.append(splitLine[0])
    if cellID in excludedCellsList:
        newlines = copy(lines)
        iLineOfInterest = excludedCellsList.index(cellID)
        if motive not in newlines[iLineOfInterest][:-1].split(','):
            newlines[iLineOfInterest] = newlines[iLineOfInterest][:-1] + ',' + motive + '\n'            
    else:
        newlines = copy(lines)
        newlines.append('' + cellID + ',' + motive + '\n')
    f.close()
    f = open(os.path.join(cp.DirRepoExp, 'ExcludedCells.txt'), 'w')
    f.writelines(newlines)
    
def getExcludedCells():
    f = open(os.path.join(cp.DirRepoExp, 'ExcludedCells.txt'), 'r')
    lines = f.readlines()
    nLines = len(lines)
    excludedCellsDict = {}
    for iLine in range(nLines):
        line = lines[iLine]
        splitLine = line[:-1].split(',')
        excludedCellsDict[splitLine[0]] = splitLine[1:]
    return(excludedCellsDict)

# %% (2) GlobalTables functions

# %%% (2.1) Exp conditions

# See UtilityFunctions.py

# %%% (2.2) Constant Field experiments

listColumnsCtField = ['date','cellName','cellID','manipID',\
                      'duration','medianRawB','medianThickness',\
                      '1stDThickness','9thDThickness','fluctuAmpli',\
                      'R2_polyFit','validated']

def analyseTimeSeries_ctField(f, tsDf, expDf):
    results = {}
    
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]
    # Deal with the asymmetric pair case : the diameter can be for instance 4503 (float) or '4503_2691' (string)
    diameters = thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'].split('_')
    if len(diameters) == 2:
        DIAMETER = (int(diameters[0]) + int(diameters[1]))/2.
    else:
        DIAMETER = int(diameters[0])
    
    results['duration'] = np.max(tsDf['T'])
    results['medianRawB'] = np.median(tsDf.B)
    results['medianThickness'] = (1000*np.median(tsDf.D3))-DIAMETER
    results['1stDThickness'] = (1000*np.percentile(tsDf.D3, 10))-DIAMETER
    results['9thDThickness'] = (1000*np.percentile(tsDf.D3, 90))-DIAMETER
    results['fluctuAmpli'] = (results['9thDThickness'] - results['1stDThickness'])
    results['validated'] = (results['1stDThickness'] > 0)

    
    # R2 polyfit to see the regularity. Order = 5 !
    X, Y = tsDf['T'], tsDf['D3']
    p, residuals, rank, singular_values, rcond = np.polyfit(X, Y, deg=5, full=True)
    Y2 = np.zeros(len(X))
    for i in range(len(X)):
        deg = len(p)-1
        for k in range(deg+1):
            Y2[i] += p[k]*(X[i]**(deg-k))
    results['R2_polyFit'] = ufun.get_R2(Y, Y2)
    return(results)



def createDataDict_ctField(list_ctFieldFiles):
    tableDict = {}
    tableDict['date'], tableDict['cellName'], tableDict['cellID'], tableDict['manipID'] = [], [], [], []
    tableDict['duration'], tableDict['medianRawB'], tableDict['medianThickness'] = [], [], []
    tableDict['1stDThickness'], tableDict['9thDThickness'], tableDict['fluctuAmpli'] = [], [], []
    tableDict['R2_polyFit'], tableDict['validated'] = [], []
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    for f in list_ctFieldFiles:
        split_f = f.split('_')
        tableDict['date'].append(split_f[0])
        tableDict['cellName'].append(split_f[1] + '_' + split_f[2] + '_' + split_f[3])
        tableDict['cellID'].append(split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3])
        tableDict['manipID'].append(split_f[0] + '_' + split_f[1])
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDf = pd.read_csv(tS_DataFilePath, ';')
        current_resultDict = analyseTimeSeries_ctField(f, current_tsDf, expDf)
        for k in current_resultDict.keys():
            tableDict[k].append(current_resultDict[k])
    return(tableDict)

# def computeGlobalTable_meca(task = 'fromScratch', fileName = 'Global_MecaData', save = False, PLOT = False, \
#                             source = 'Matlab', listColumnsMeca=listColumnsMeca):
#     """
#     Compute the GlobalTable_meca from the time series data files.
#     Option task='fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.
#     Option task='updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new fils.
#     listColumnsMeca have to contain all the fields of the table that will be constructed.
#     """
#     top = time.time()
    
# #     list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
# #                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
# #                       and ('R40' in f))] # Change to allow different formats in the future
    
#     suffixPython = '_PY'
#     if source == 'Matlab':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and ('R40' in f) and not (suffixPython in f))]
        
#     elif source == 'Python':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and (('R40' in f) or ('L40' in f)) and (suffixPython in f))]
#         # print(list_mecaFiles)

def computeGlobalTable_ctField(task = 'fromScratch', fileName = 'Global_CtFieldData', save = False,
                               source = 'Matlab'):
    """
    Compute the GlobalTable_ctField from the time series data files.
    > Option task='fromScratch' will analyse all the time series data files and construct 
    a new GlobalTable from them regardless of the existing GlobalTable.
    > Option task='updateExisting' will open the existing GlobalTable and determine 
    which of the time series data files are new ones, and will append the existing GlobalTable 
    with the data analysed from those new files.
    > Else, having task= a date, a cellID or a manipID will create a globalTable with this source only.
    """
    ctFieldFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                                  if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                                      and ('thickness' in f))]
        
#     print(ctFieldFiles)

    suffixPython = '_PY'
    if source == 'Matlab':
        ctFieldFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and ('thickness' in f) and not (suffixPython in f))]
        
    elif source == 'Python':
        ctFieldFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and ('thickness' in f) and (suffixPython in f))]
        # print(list_mecaFiles)


    if task == 'fromScratch':
        # create a dict containing the data
        tableDict = createDataDict_ctField(ctFieldFiles) # MAIN SUBFUNCTION
        # create the table
        CtField_DF = pd.DataFrame(tableDict)
        
    elif task == 'updateExisting':
        # get existing table
        try:
            savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
            existing_CtField_DF = pd.read_csv(savePath, sep=';')
            for c in existing_CtField_DF.columns:
                if 'Unnamed' in c:
                    existing_CtField_DF = existing_CtField_DF.drop([c], axis=1)
        except:
            print('No existing table found')
        # find which of the time series files are new
        new_ctFieldFiles = []
        for f in ctFieldFiles:
            split_f = f.split('_')
            currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
            if currentCellID not in existing_CtField_DF.cellID.values:
                new_ctFieldFiles.append(f)
        new_tableDict = createDataDict_ctField(new_ctFieldFiles) # MAIN SUBFUNCTION
        # create the table with new data
        new_CtField_DF = pd.DataFrame(new_tableDict)
        # fuse the two
        new_CtField_DF.index += existing_CtField_DF.shape[0]
        CtField_DF = pd.concat([existing_CtField_DF, new_CtField_DF])
        
    else: # If task is neither 'fromScratch' nor 'updateExisting'
    # Then task can be a substring that can be in some timeSeries file !
    # It will create a table with only these files, WITH OR WITHOUT SAVING IT !
    # And it can plot figs from it.
        # save = False
        new_ctFieldFiles = []
        for f in new_ctFieldFiles:
            split_f = f.split('_')
            currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
            if task in currentCellID:
                new_ctFieldFiles.append(f)
        # create the dict with new data
        new_tableDict = createDataDict_ctField(new_ctFieldFiles) # MAIN SUBFUNCTION
        # create the dataframe from it
        CtField_DF = pd.DataFrame(new_tableDict)
    
    dateExemple = CtField_DF.loc[CtField_DF.index[0],'date']
    if re.match(ufun.dateFormatExcel, dateExemple):
        CtField_DF.loc[:,'date'] = CtField_DF.loc[:,'date'].apply(lambda x: x.split('/')[0] + '-' + x.split('/')[1] + '-' + x.split('/')[2][2:])
    
    for c in CtField_DF.columns:
        if 'Unnamed' in c:
            CtField_DF = CtField_DF.drop([c], axis=1)
    
    if save:
        saveName = fileName + '.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        CtField_DF.to_csv(savePath, sep=';')
        
    return(CtField_DF)



def getGlobalTable_ctField(fileName = 'Global_CtFieldData'):
    try:
        savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
        CtField_DF = pd.read_csv(savePath, sep=';')
        for c in CtField_DF.columns:
            if 'Unnamed' in c:
                CtField_DF = CtField_DF.drop([c], axis=1)
        print('Extracted a table with ' + str(CtField_DF.shape[0]) + ' lines and ' + str(CtField_DF.shape[1]) + ' columns.')
        
    except:
        print('No existing table found')
        
    dateExemple = CtField_DF.loc[CtField_DF.index[0],'date']
    if re.match(ufun.dateFormatExcel, dateExemple):
        print('dates corrected')
        CtField_DF.loc[:,'date'] = CtField_DF.loc[:,'date'].apply(lambda x: x.split('/')[0] + '-' + x.split('/')[1] + '-' + x.split('/')[2][2:])
#         mecaDF['ManipID'] = mecaDF['ExpDay'] + '_' + mecaDF['CellName'].apply(lambda x: x.split('_')[0])
    return(CtField_DF)

# %%% (2.3) Compressions experiments

#### Workflow
# * analyseTimeSeries_meca() analyse 1 file and return the dict (with the results of the analysis)
# * createMecaDataDict() call the previous function on the given list of files and concatenate the results
# * computeGlobalTable_meca() call the previous function and convert the dict to a DataFrame

#### Set dictColumnsMeca
dictColumnsMeca = {'date':'',
                   'cellName':'',
                   'cellID':'',
                   'manipID':'',
                   'compNum':np.nan,
                   'compDuration':'',
                   'compStartTime':np.nan,
                   'compAbsStartTime':np.nan,
                   'compStartTimeThisDay':np.nan,
                   'initialThickness':np.nan,
                   'minThickness':np.nan,
                   'maxIndent':np.nan,
                   'previousThickness':np.nan,
                   'surroundingThickness':np.nan,
                   'surroundingDx':np.nan,
                   'surroundingDz':np.nan,
                   'validatedThickness':np.nan, 
                   'jumpD3':np.nan,
                   'minForce':np.nan, 
                   'maxForce':np.nan, 
                   'minStress':np.nan, 
                   'maxStress':np.nan, 
                   'minStrain':np.nan, 
                   'maxStrain':np.nan,
                   'ctFieldThickness':np.nan,
                   'ctFieldFluctuAmpli':np.nan,
                   'ctFieldMinThickness':np.nan,
                   'ctFieldMaxThickness':np.nan,
                   'ctFieldVarThickness':np.nan,
                   'ctFieldDX':np.nan,
                   'ctFieldDZ':np.nan,
                   'bestH0':np.nan,
                   'error_bestH0':False,
                   'H0_Chadwick15':np.nan, 
                   'H0_Dimitriadis15':np.nan, 
                   'H0Chadwick':np.nan,
                   'EChadwick':np.nan,
                   'R2Chadwick':np.nan,
                   'EChadwick_CIWidth':np.nan,
                   'hysteresis':np.nan,
                   'validatedFit_Chadwick':False,
                   'critFit':'',
                   'comments':''
                   }

#### SETTINGS !

# H0_bestMethod
H0_bestMethod = 'H0_Chadwick15'

# Fit Selection R2 & Chi2
dictSelectionCurve = {'R2' : 0.6, 'Chi2' : 10, 'Error' : 0.02}

# Regions definitions - OPTION 5 : Many range width
fitC =  np.array([S for S in range(100, 1150, 50)])
fitW = [100, 150, 200]
# fitW = [100, 150, 200, 250, 300]


#### Compute the regions

fitCenters = np.array([[int(S) for S in fitC] for w in fitW]).flatten()
fitWidth = np.array([[int(w) for S in fitC] for w in fitW]).flatten()

fitMin = np.array([[int(S-(w/2)) for S in fitC] for w in fitW]).flatten()
fitMax = np.array([[int(S+(w/2)) for S in fitC] for w in fitW]).flatten()

fitCenters, fitWidth = fitCenters[fitMin>0], fitWidth[fitMin>0]
fitMin, fitMax = fitMin[fitMin>0], fitMax[fitMin>0]

regionFitsNames = ['S='  + str(fitCenters[ii]) + '+/-' + str(int(fitWidth[ii]//2)) for ii in range(len(fitCenters))]

fitsToPlot = [regionFitsNames[ii] for ii in range(len(fitC), 2*len(fitC), 2)]
mask_selectFitsToPlot = np.array(list(map(lambda x : x in fitsToPlot, regionFitsNames)))

dictColumnsRegionFit = {'regionFitNames' : '', 
                        'K' : np.nan, 
                        'K_CIW' : np.nan,
                        'R2' : np.nan,  
                        'Npts' : np.nan, 
                        'fitError' : True, 
                        'validatedFit' : False
                        } 



#### Complete dictColumnsMeca

for rfn in regionFitsNames:
     d = {'K_'+rfn:np.nan, 
          'K_CIW_'+rfn:np.nan, 
          'R2_'+rfn:np.nan, 
          'Npts_'+rfn:np.nan,
          'fitError_'+rfn:True, 
          'validatedFit_'+rfn:False
          }
     dictColumnsMeca = {**dictColumnsMeca, **d}
     
    


     

def compressionFitChadwick(hCompr, fCompr, DIAMETER):
    
    error = False
    
    def chadwickModel(h, E, H0):
        R = DIAMETER/2
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        R = DIAMETER/2
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    # some initial parameter values - must be within bounds
    initH0 = max(hCompr) # H0 ~ h_max
    initE = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    # initH0, initE = initH0*(initH0>0), initE*(initE>0)
    
    initialParameters = [initE, initH0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, 0)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]

    try:
        params, covM = curve_fit(inversedChadwickModel, fCompr, hCompr, initialParameters, bounds = parameterBounds)

        # Previously I fitted with y=F and x=H, but it didn't work so well cause H(t) isn't monotonous:
        # params, covM = curve_fit(chadwickModel, hCompr, fCompr, initialParameters, bounds = parameterBounds)
        # Fitting with the 'inverse Chadwick model', with y=H and x=F is more convenient

        E, H0 = params
        hPredict = inversedChadwickModel(fCompr, E, H0)
        err = dictSelectionCurve['Error']
        
        comprMat = np.array([hCompr, fCompr]).T
        comprMatSorted = comprMat[comprMat[:, 0].argsort()]
        hComprSorted, fComprSorted = comprMatSorted[:, 0], comprMatSorted[:, 1]
        fPredict = chadwickModel(hComprSorted, E, H0)
        
        # Stress and strain
        deltaCompr = (H0 - hCompr)/1000 # µm
        stressCompr = fCompr / (np.pi * (DIAMETER/2000) * deltaCompr)
        strainCompr = deltaCompr / (3*(H0/1000)) 
        strainPredict = stressCompr / (E*1e6) #((H0 - hPredict)/1000) / (3*(H0/1000))
        
        # residuals_h = hCompr-hPredict
        # residuals_f = fComprSorted-fPredict

        alpha = 0.975
        dof = len(fCompr)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(hCompr, hPredict)
        
        Chi2 = ufun.get_Chi2(strainCompr, strainPredict, dof, err)

        varE = covM[0,0]
        seE = (varE)**0.5
        E, seE = E*1e6, seE*1e6
        confIntE = [E-q*seE, E+q*seE]
        confIntEWidth = 2*q*seE

        varH0 = covM[1,1]
        seH0 = (varH0)**0.5
        confIntH0 = [H0-q*seH0, H0+q*seH0]
        confIntH0Width = 2*q*seH0
        
        
    except:
        error = True
        E, H0, hPredict, R2, Chi2, confIntE, confIntH0 = -1, -1, np.ones(len(hCompr))*(-1), -1, -1, [-1,-1], [-1,-1]
    
    return(E, H0, hPredict, R2, Chi2, confIntE, confIntH0, error)


def compressionFitDimitriadis(hCompr, fCompr, DIAMETER, order = 2):
    
    error = False
    
    v = 0
    
    def dimitriadisModel(h, E, H0):
        
        R = DIAMETER/2
        delta = H0-h
        X = np.sqrt(R*delta)/h
        ks = ufun.getDimitriadisCoefs(v, order)
        
        poly = np.zeros_like(X)
        
        for i in range(order+1):
            poly = poly + ks[i] * X**i
            
        F = ((4 * E * R**0.5 * delta**1.5)/(3 * (1 - v**2))) * poly
        return(F)

    # some initial parameter values - must be within bounds
    # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    initE = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) 
    # H0 ~ h_max
    initH0 = max(hCompr) 

    # initH0, initE = initH0*(initH0>0), initE*(initE>0)
    
    initialParameters = [initE, initH0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, max(hCompr))
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]

    try:
        
        params, covM = curve_fit(dimitriadisModel, hCompr, fCompr, p0=initialParameters, bounds = parameterBounds)
        E, H0 = params
        fPredict = dimitriadisModel(hCompr, E, H0)
        err = 100
        
        comprMat = np.array([hCompr, fPredict]).T
        comprMatSorted = comprMat[comprMat[:, 0].argsort()]
        hComprSorted, fPredictSorted = comprMatSorted[:, 0], comprMatSorted[:, 1]

        alpha = 0.975
        dof = len(fCompr)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(fCompr, fPredict)
        
        Chi2 = ufun.get_Chi2(fCompr, fPredict, dof, err)
        
        varE = covM[0,0]
        seE = (varE)**0.5
        E, seE = E*1e6, seE*1e6
        confIntE = [E-q*seE, E+q*seE]
        confIntEWidth = 2*q*seE

        varH0 = covM[1,1]
        seH0 = (varH0)**0.5
        confIntH0 = [H0-q*seH0, H0+q*seH0]
        confIntH0Width = 2*q*seH0
        
        
    except:
        error = True
        E, H0, fPredict, R2, Chi2, confIntE, confIntH0 = -1, -1, np.ones(len(hCompr))*(-1), -1, -1, [-1,-1], [-1,-1]
    
    return(E, H0, fPredict, R2, Chi2, confIntE, confIntH0, error)


    

def compressionFitChadwick_StressStrain(hCompr, fCompr, H0, DIAMETER):
    
    error = False
    
    # def chadwickModel(h, E, H0):
    #     R = DIAMETER/2
    #     f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    #     return(f)

    # def inversedChadwickModel(f, E, H0):
    #     R = DIAMETER/2
    #     h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    #     return(h)
    
    def computeStress(f, h, H0):
        R = DIAMETER/2000
        delta = (H0 - h)/1000
        stress = f / (np.pi * R * delta)
        return(stress)
        
    def computeStrain(h, H0):
        delta = (H0 - h)/1000
        strain = delta / (3 * (H0/1000))
        return(strain)
    
    def constitutiveRelation(strain, K, stress0):
        stress = (K * strain) + stress0
        return(stress)
    
    def inversedConstitutiveRelation(stress, K, strain0):
        strain = (stress / K) + strain0
        return(strain)

    # some initial parameter values - must be within bounds
    initK = (3*max(hCompr)*max(fCompr))/(np.pi*(DIAMETER/2)*(max(hCompr)-min(hCompr))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
    init0 = 0
    
    initialParameters = [initK, init0]

    # bounds on parameters - initial parameters must be within these
    lowerBounds = (0, -np.Inf)
    upperBounds = (np.Inf, np.Inf)
    parameterBounds = [lowerBounds, upperBounds]


    try:
        strainCompr = computeStrain(hCompr, H0)
        stressCompr = computeStress(fCompr, hCompr, H0)
        
        params, covM = curve_fit(inversedConstitutiveRelation, stressCompr, strainCompr, initialParameters, bounds = parameterBounds)

        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stressCompr, K, strain0)
        err = dictSelectionCurve['Error']

        alpha = 0.975
        dof = len(stressCompr)-len(params)
        
        R2 = ufun.get_R2(strainCompr, strainPredict)
        
        Chi2 = ufun.get_Chi2(strainCompr, strainPredict, dof, err)

        varK = covM[0,0]
        seK = (varK)**0.5
        
        q = st.t.ppf(alpha, dof) # Student coefficient
        # K, seK = K*1e6, seK*1e6
        confIntK = [K-q*seK, K+q*seK]
        confIntKWidth = 2*q*seK
        
        
    except:
        error = True
        K, strainPredict, R2, Chi2, confIntK = -1, np.ones(len(strainCompr))*(-1), -1, -1, [-1,-1]
    
    return(K, strainPredict, R2, Chi2, confIntK, error)



def fitH0_allMethods(hCompr, fCompr, DIAMETER):
    dictH0 = {}
    
    # findH0_E, H0_Chadwick15, findH0_hPredict, findH0_R2, findH0_Chi2, findH0_confIntE, findH0_confIntH0, findH0_fitError
    Chadwick15_resultTuple = compressionFitChadwick(hCompr[:15], fCompr[:15], DIAMETER)
    H0_Chadwick15 = Chadwick15_resultTuple[1]
    dictH0['H0_Chadwick15'] = H0_Chadwick15
    dictH0['H0_Chadwick15_resTuple'] = Chadwick15_resultTuple
    
    
    
    Dimitriadis15_resultTuple = compressionFitDimitriadis(hCompr[:15], fCompr[:15], DIAMETER)
    H0_Dimitriadis15 = Dimitriadis15_resultTuple[1]
    dictH0['H0_Dimitriadis15'] = H0_Dimitriadis15
    dictH0['H0_Dimitriadis15_resTuple'] = Dimitriadis15_resultTuple
    
    return(dictH0)



def analyseTimeSeries_meca(f, tsDF, expDf, dictColumnsMeca, task, PLOT, saveAnalysedTs = False):
    
    print(gs.BLUE + f + gs.NORMAL)
    plt.ioff()
    
    #### (0) Import experimental infos
    tsDF.dx, tsDF.dy, tsDF.dz, tsDF.D2, tsDF.D3 = tsDF.dx*1000, tsDF.dy*1000, tsDF.dz*1000, tsDF.D2*1000, tsDF.D3*1000
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]

    # Deal with the asymmetric pair case : the diameter can be for instance 4503 (float) or '4503_2691' (string)
    diameters = thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'].split('_')
    if len(diameters) == 2:
        DIAMETER = (int(diameters[0]) + int(diameters[1]))/2.
    else:
        DIAMETER = int(diameters[0])
    
    EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])
    Ncomp = max(tsDF['idxAnalysis'])
    
    # Field infos
    normalField = thisExpDf.at[thisExpDf.index.values[0], 'normal field']
    normalField = int(normalField)
    if 'compression' in EXPTYPE:
        compField = thisExpDf.at[thisExpDf.index.values[0], 'ramp field'].split('_')
        minCompField = float(compField[0])
        maxCompField = float(compField[1])
    
    # Loop structure infos
    loopStruct = thisExpDf.at[thisExpDf.index.values[0], 'loop structure'].split('_')
    nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
    if 'compression' in EXPTYPE:
        loop_totalSize = int(loopStruct[0])
        if len(loopStruct) >= 2:
            loop_rampSize = int(loopStruct[1])
        else:
            loop_rampSize = 0
        if len(loopStruct) >= 3:
            loop_excludedSize = int(loopStruct[2])
        else:
            loop_excludedSize = 0
        loop_ctSize = int((loop_totalSize - (loop_rampSize+loop_excludedSize))/nUplet)
        
    
    # Initialize the results
    results = {}        
    for k in dictColumnsMeca.keys():
        results[k] = [dictColumnsMeca[k] for m in range(Ncomp)]



    #### (1) Get global values
    # These values are computed once for the whole cell D3 time series, but since the table has 1 line per compression, 
    # that same value will be put in the table for each line corresponding to that cell
    ctFieldH = (tsDF.loc[tsDF['idxAnalysis'] == 0, 'D3'].values - DIAMETER)
    ctFieldDX = np.median(tsDF.loc[tsDF['idxAnalysis'] == 0, 'dx'].values - DIAMETER)
    ctFieldDZ = np.median(tsDF.loc[tsDF['idxAnalysis'] == 0, 'dz'].values)
    ctFieldThickness    = np.median(ctFieldH)
    ctFieldMinThickness = np.min(ctFieldH)
    ctFieldMaxThickness = np.max(ctFieldH)
    ctFieldVarThickness = np.var(ctFieldH)
    ctFieldFluctuAmpli  = np.percentile(ctFieldH, 90) - np.percentile(ctFieldH,10)

    
    #### Save stress-strain [1/3] 
    # Export a file "Timeseries_stress-strain"
    if saveAnalysedTs:
        Nrows = tsDF.shape[0]
        ts_H0 = np.full(Nrows, np.nan)
        ts_stress = np.full(Nrows, np.nan)
        ts_strain = np.full(Nrows, np.nan)
    
    #### PLOT [1/4]
    # First part of the plot [mainly ax1 and ax1bis]
    if PLOT:
        # 1st plot - fig1 & ax1, is the F(t) curves with the different compressions colored depending of the analysis success
        fig1, ax1 = plt.subplots(1,1,figsize=(tsDF.shape[0]*(1/100),5))
        color = 'blue'
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('h (nm)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        nColsSubplot = 5
        nRowsSubplot = ((Ncomp-1) // nColsSubplot) + 1
        
        # 2nd plot - fig2 & ax2, gather all the F(h) curves, and will be completed later in the code
        fig2, ax2 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        
        # 3rd plot - fig3 & ax3, gather all the stress-strain curves, and will be completed later in the code
        fig3, ax3 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        
        # 4th plot - fig4 & ax4, gather all the F(h) curves with local fits, and will be completed later in the code
        fig4, ax4 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        alreadyLabeled4 = []
        
        # 5th plot - fig5 & ax5, gather all the stress-strain curves with local fits, and will be completed later in the code
        fig5, ax5 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        alreadyLabeled5 = []
        
        # 6th plot - fig6 & ax6
        fig6, ax6 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))
        
        # 7th plot - fig7 & ax7, gather all the "small elements", and will be completed later in the code
        fig7, ax7 = plt.subplots(nRowsSubplot,nColsSubplot,figsize=(3*nColsSubplot,3*nRowsSubplot))



    for i in range(Ncomp):#Ncomp+1):

        #### (1) Identifiers
        results['date'][i] = ufun.findInfosInFileName(f, 'date')
        results['cellName'][i] = ufun.findInfosInFileName(f, 'cellName')
        results['cellID'][i] = ufun.findInfosInFileName(f, 'cellID')
        results['manipID'][i] = ufun.findInfosInFileName(f, 'manipID')
        
        currentCellID = results['cellID'][i]
        
        #### Jump correction
        if EXPTYPE == 'compressionsLowStart' or normalField == minCompField:
            colToCorrect = ['dx', 'dy', 'dz', 'D2', 'D3']
            maskCompAndPrecomp = np.abs(tsDF['idxAnalysis']) == i+1
            iStart = ufun.findFirst(np.abs(tsDF['idxAnalysis']), i+1)
            for c in colToCorrect:
                jump = np.mean(tsDF[c].values[iStart:iStart+3]) - tsDF[c].values[iStart-1]
                tsDF.loc[maskCompAndPrecomp, c] -= jump
                if c == 'D3':
                    D3corrected = True
                    jumpD3 = jump
        else:
            D3corrected = False
            jumpD3 = 0
            
        #### (2) Segment the compression n°i
        thisCompDf = tsDF.loc[tsDF['idxAnalysis'] == i+1,:]
        iStart = (ufun.findFirst(tsDF['idxAnalysis'], i+1))
        iStop = iStart+thisCompDf.shape[0]

        # Easy-to-get parameters
        results['compNum'][i] = i+1
        results['compDuration'][i] = thisExpDf.at[thisExpDf.index.values[0], 'compression duration']
        results['compStartTime'][i] = thisCompDf['T'].values[0]
        results['compAbsStartTime'][i] = thisCompDf['Tabs'].values[0]
        results['compStartTimeThisDay'][i] = thisCompDf['Tabs'].values[0]

        listB = thisCompDf.B.values
        
        # Test to check if most of the compression have not been deleted due to bad image quality 
        highBvalues = (listB > (maxCompField + minCompField)/2)
        N_highBvalues = np.sum(highBvalues)
        testHighVal = (N_highBvalues > 20)

        # Test to check if the range of B field is large enough
        minB, maxB = min(listB), max(listB)
        testRangeB = ((maxB-minB) > 0.7*(maxCompField - minCompField))
        thresholdB = (maxB-minB)/50
        thresholdDeltaB = (maxB-minB)/400

        #### Is the curve ok to analyse ?
        doThisCompAnalysis = testHighVal and testRangeB # Some criteria can be added here

        if doThisCompAnalysis:
            #### (3) Inside the compression n°i, delimit the compression and relaxation phases

            # Delimit the start of the increase of B (typically the moment when the field decrease from 5 to 3)
            # and the end of its decrease (typically when it goes back from 3 to 5)
            
            try:
                # Correct for bugs in the B data
                if 'compressions' in EXPTYPE:
                    for k in range(1,len(listB)):
                        B = listB[k]
                        if B > 1.25*maxCompField:
                            listB[k] = listB[k-1]

                offsetStart, offsetStop = 0, 0
                minB, maxB = min(listB), max(listB)
                thresholdB = (maxB-minB)/50
                thresholdDeltaB = (maxB-minB)/400 # NEW CONDITION for the beginning of the compression : 
                # remove the first points where the steps in B are very very small

                k = 0
                while (listB[k] < minB+thresholdB) or (listB[k+1]-listB[k] < thresholdDeltaB):
                    offsetStart += int((listB[k] < minB+thresholdB) or ((listB[k+1]-listB[k]) < thresholdDeltaB))
                    k += 1

                k = 0
                while (listB[-1-k] < minB+thresholdB):
                    offsetStop += int(listB[-1-k] < minB+thresholdB)
                    k += 1

                jStart = offsetStart # Beginning of compression
                jMax = np.argmax(thisCompDf.B) # End of compression, beginning of relaxation
                jStop = thisCompDf.shape[0] - offsetStop # End of relaxation
            
            except:
                print(listB)
                print(testRangeB, thresholdB, thresholdDeltaB)
            
            # Four arrays
            hCompr = (thisCompDf.D3.values[jStart:jMax+1] - DIAMETER)
            hRelax = (thisCompDf.D3.values[jMax+1:jStop] - DIAMETER)
            fCompr = (thisCompDf.F.values[jStart:jMax+1])
            fRelax = (thisCompDf.F.values[jMax+1:jStop])
            

            # Refinement of the compression delimitation.
            # Remove the 1-2 points at the begining where there is just the viscous relaxation of the cortex
            # because of the initial decrease of B and the cortex thickness increases.
            offsetStart2 = 0
            k = 0
            while (k<len(hCompr)-10) and (hCompr[k] < np.max(hCompr[k+1:min(k+10, len(hCompr))])):
                offsetStart2 += 1
                k += 1
            # Better compressions arrays
            hCompr = hCompr[offsetStart2:]
            fCompr = fCompr[offsetStart2:]
            
            
            
            # Get the points of constant field preceding and surrounding the current compression
            # Ex : if the labview code was set so that there is 6 points of ct field before and after each compression,
            # previousPoints will contains D3[iStart-12:iStart]
            # surroundingPoints will contains D3[iStart-6:iStart] and D3[iStop:iStop+6]
            previousPoints = (tsDF.D3.values[max(0,iStart-(loop_ctSize)):iStart]) - DIAMETER
            surroundingPoints = np.concatenate([tsDF.D3.values[max(0,iStart-(loop_ctSize//2)):iStart],tsDF.D3.values[iStop:iStop+(loop_ctSize//2)]]) - DIAMETER
            surroundingPointsX = np.concatenate([tsDF.dx.values[max(0,iStart-(loop_ctSize//2)):iStart],tsDF.dx.values[iStop:iStop+(loop_ctSize//2)]]) - DIAMETER
            surroundingPointsZ = np.concatenate([tsDF.dz.values[max(0,iStart-(loop_ctSize//2)):iStart],tsDF.dz.values[iStop:iStop+(loop_ctSize//2)]])
            
            # Parameters relative to the thickness ( = D3-DIAMETER)            
            results['initialThickness'][i] = np.mean(hCompr[0:3])
            results['minThickness'][i] = np.min(hCompr)
            results['maxIndent'][i] = results['initialThickness'][i] - results['minThickness'][i]
            results['previousThickness'][i] = np.median(previousPoints)
            results['surroundingThickness'][i] = np.median(surroundingPoints)
            results['surroundingDx'][i] = np.median(surroundingPointsX)
            results['surroundingDz'][i] = np.median(surroundingPointsZ)
            results['ctFieldDX'][i] = ctFieldDX
            results['ctFieldDZ'][i] = ctFieldDZ
            results['ctFieldThickness'][i] = ctFieldThickness
            results['ctFieldFluctuAmpli'][i] = ctFieldFluctuAmpli
            results['ctFieldMinThickness'][i] = ctFieldMinThickness
            results['ctFieldMaxThickness'][i] = ctFieldMaxThickness
            results['ctFieldVarThickness'][i] = ctFieldVarThickness
            results['jumpD3'][i] = jumpD3

            validatedThickness = np.min([results['initialThickness'][i],results['minThickness'][i],
                                         results['previousThickness'][i],results['surroundingThickness'][i],
                                         results['ctFieldThickness'][i]]) > 0
            
            results['validatedThickness'][i] = validatedThickness
            results['minForce'][i] = np.min(fCompr)
            results['maxForce'][i] = np.max(fCompr)

            
    

            #### (4) Fit with Chadwick model of the force-thickness curve
            
            #### Classic, all curve, Chadwick fit
            E, H0, hPredict, R2, Chi2, confIntE, confIntH0, fitError = compressionFitChadwick(hCompr, fCompr, DIAMETER) # IMPORTANT SUBFUNCTION

            R2CRITERION = dictSelectionCurve['R2']
            CHI2CRITERION = dictSelectionCurve['Chi2']
            critFit = 'R2 > ' + str(R2CRITERION)
            
            # results['critFit'].append(critFit)
            results['critFit'][i] = critFit
            
            validatedFit = ((R2 > R2CRITERION) and (Chi2 < CHI2CRITERION))

            if not fitError:
                # results['validatedFit_Chadwick'].append(validatedFit)
                results['validatedFit_Chadwick'][i] = validatedFit
                if validatedFit:
                    # results['comments'].append('ok')
                    results['comments'][i] = 'ok'
                else:
                    # results['comments'].append('R2 < ' + str(R2CRITERION))
                    results['comments'][i] = 'R2 < ' + str(R2CRITERION)

                confIntEWidth = abs(confIntE[0] - confIntE[1])
                
                results['H0Chadwick'][i] = H0
                results['EChadwick'][i] = E
                results['R2Chadwick'][i] = R2
                results['EChadwick_CIWidth'][i] = confIntEWidth
                
            elif fitError:
                results['comments'][i] = 'fitFailure'
                
                
            #### (4.0) Find the best H0
            dictH0 = fitH0_allMethods(hCompr, fCompr, DIAMETER)
            
            H0_Chadwick15 = dictH0['H0_Chadwick15']
            H0_Dimitriadis15 = dictH0['H0_Dimitriadis15']
            try:
                results['H0_Chadwick15'][i] = H0_Chadwick15
                results['H0_Dimitriadis15'][i] = H0_Dimitriadis15
            except:
                pass
            
            bestH0 = dictH0[H0_bestMethod]
            resTuple_bestH0 = dictH0[H0_bestMethod + '_resTuple']
            error_bestH0 = resTuple_bestH0[-1]
            results['error_bestH0'][i] = error_bestH0

            if max(hCompr) > bestH0:
                error_bestH0 = True          


            
            if not error_bestH0:
                
                #### (4.1) Compute stress and strain based on the best H0
                
                results['bestH0'][i] = bestH0
                
                #### Stress-strain computation
                deltaCompr = (bestH0 - hCompr)/1000
                stressCompr = fCompr / (np.pi * (DIAMETER/2000) * deltaCompr)
                strainCompr = deltaCompr / (3*(bestH0/1000))
                # trueStrainCompr = np.log((H0_Chadwick15)/(hCompr*(hCompr>0)))                
                
                results['minStress'][i] = np.min(stressCompr)
                results['maxStress'][i] = np.max(stressCompr)
                results['minStrain'][i] = np.min(strainCompr)
                results['maxStrain'][i] = np.max(strainCompr)              
                
                
                #### (4.2) Fits on specific regions of the curve
                    
                N_Fits = len(regionFitsNames)
                
                dictRegionFit = {}
                for k in dictColumnsRegionFit.keys():
                    dictRegionFit[k] = [dictColumnsRegionFit[k] for m in range(N_Fits)]
                # dictColumnsRegionFit = {'regionFitNames' : '', 'K' : np.nan, 'K_CIW' : np.nan, 'R2' : np.nan,  
                #                         'Npts' : np.nan, 'fitError' : True, 'validatedFit' : False} 
                
                all_stressMasks_region = np.zeros((N_Fits, len(stressCompr)), dtype = bool)
                list_strainPredict_fitToPlot = [[] for kk in range(len(fitsToPlot))]                
                
                for kk in range(N_Fits):
                    ftP = regionFitsNames[kk]
                    lowS, highS = int(fitMin[kk]), int(fitMax[kk])
                    all_stressMasks_region[kk,:] = ((stressCompr > lowS) & (stressCompr < highS))
                
                for ii in range(N_Fits):
                    regionFitName = regionFitsNames[ii]
                    mask_region = all_stressMasks_region[ii]
                    Npts_region = np.sum(mask_region)
                    
                    if Npts_region > 5:
                        fCompr_region = fCompr[mask_region]
                        hCompr_region = hCompr[mask_region]
                                                
                        K_region, strainPredict_region, R2_region, Chi2_region, confIntK_region, fitError_region = \
                            compressionFitChadwick_StressStrain(hCompr_region, fCompr_region, bestH0, DIAMETER)
                            
                        confIntWidthK_region = np.abs(confIntK_region[0] - confIntK_region[1])
            
                    else:
                        fitError_region = True
                        
                        
                    if (regionFitName in fitsToPlot) and not fitError_region:
                        kk = np.argmax(np.array(fitsToPlot) == regionFitName)
                        list_strainPredict_fitToPlot[kk] = strainPredict_region
                    
                    if not fitError_region:
                        R2CRITERION = dictSelectionCurve['R2']
                        CHI2CRITERION = dictSelectionCurve['Chi2']
                        validatedFit_region = ((R2_region > R2CRITERION) and 
                                                (Chi2_region < CHI2CRITERION))
                        
                        # Fill dictRegionFit (reinitialized for each cell)
                        dictRegionFit['regionFitNames'][ii] = regionFitName
                        dictRegionFit['K'][ii] = K_region
                        dictRegionFit['K_CIW'][ii] = confIntWidthK_region
                        dictRegionFit['R2'][ii] = R2_region
                        dictRegionFit['Npts'][ii] = Npts_region
                        dictRegionFit['fitError'][ii] = fitError_region
                        dictRegionFit['validatedFit'][ii] = validatedFit_region
                        # dictColumnsRegionFit = {'regionFitNames' : '', 'K' : np.nan, 'K_CIW' : np.nan, 'R2' : np.nan,  
                        #                         'Npts' : np.nan, 'fitError' : True, 'validatedFit' : False} 
                        
                        # Fill results (memorize everything)
                        rFN = regionFitName
                        results['K_'+rFN][i] = K_region
                        results['K_CIW_'+rFN][i] = confIntWidthK_region                        
                        results['R2_'+rFN][i] = R2_region
                        results['Npts_'+rFN][i] = Npts_region
                        results['fitError_'+rFN][i] = fitError_region
                        results['validatedFit_'+rFN][i] = validatedFit_region
                        # d = {'K_'+rfn:np.nan, 'K_CIW_'+rfn:np.nan, 'R2_'+rfn:np.nan, 
                        #      'Npts_'+rfn:np.nan, 'fitError_'+rfn:True, 'validatedFit_'+rfn:False}

                for k in dictRegionFit.keys():
                    dictRegionFit[k] = np.array(dictRegionFit[k])
            
            
            
            #### PLOT [2/4]
            # Complete fig 1, 2, 3 with the results of the fit
            if PLOT:
                
                #### fig1
                if not fitError:
                    ax1.plot(thisCompDf['T'].values, thisCompDf['D3'].values-DIAMETER, color = 'chartreuse', linestyle = '-', linewidth = 1.25)
                    
                else:
                    ax1.plot(thisCompDf['T'].values, thisCompDf['D3'].values-DIAMETER, color = 'crimson', linestyle = '-', linewidth = 1.25)
                
                # Display jumpD3 >>> DISABLED
                # if jumpD3 != 0:
                #     x = np.mean(thisCompDf['T'].values)
                #     y = np.mean(thisCompDf['D3'].values-DIAMETER) * 1.3
                #     ax1.text(x, y, '{:.2f}'.format(jumpD3), ha = 'center')

                fig1.suptitle(currentCellID)
                

                #### fig2 & fig3
                colSp = (i) % nColsSubplot
                rowSp = (i) // nColsSubplot
                # ax2[i-1] with the 1 line plot
                if nRowsSubplot == 1:
                    thisAx2 = ax2[colSp]
                    thisAx3 = ax3[colSp]
                elif nRowsSubplot >= 1:
                    thisAx2 = ax2[rowSp,colSp]
                    thisAx3 = ax3[rowSp,colSp]

                thisAx2.plot(hCompr,fCompr,'b-', linewidth = 0.8)
                thisAx2.plot(hRelax,fRelax,'r-', linewidth = 0.8)
                titleText = currentCellID + '__c' + str(i+1)
                legendText = ''
                thisAx2.set_xlabel('h (nm)')
                thisAx2.set_ylabel('f (pN)')


                if not fitError:
                    legendText += 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                    thisAx2.plot(hPredict,fCompr,'k--', linewidth = 0.8, label = legendText, zorder = 2)
                    thisAx2.legend(loc = 'upper right', prop={'size': 6})
                    # if not validatedFit:
                    #     titleText += '\nNON VALIDATED'
                        
                    if not error_bestH0:
                        thisAx3.plot(stressCompr, strainCompr, 'go', ms = 3)
                        # thisAx3.plot(stressCompr, trueStrainCompr, 'bo', ms = 2)
                        thisAx3.plot([np.percentile(stressCompr,10), np.percentile(stressCompr,90)], 
                                     [np.percentile(stressCompr,10)/E, np.percentile(stressCompr,90)/E],
                                     'k--', linewidth = 1.2, label = legendText)
                        thisAx3.legend(loc = 'lower right', prop={'size': 6})
                    thisAx3.set_xlabel('Stress (Pa)')
                    thisAx3.set_ylabel('Strain')
                    
                else:
                    titleText += '\nFIT ERROR'
                    
                if not error_bestH0:
                    # Computations to display the fit of the H0_Chadwick15
                    
                    bestH0_resTuple = dictH0[H0_bestMethod + '_resTuple']
                    # (E, H0, fPredict, R2, Chi2, confIntE, confIntH0, error)
                    NbPts_bestH0 = 15
                    E_bestH0 = bestH0_resTuple[0]
                    hPredict_bestH0 = bestH0_resTuple[2]
                    
                    min_f = np.min(fCompr)
                    low_f = np.linspace(0, min_f, 20)
                    R = DIAMETER/2
                    low_h = bestH0 - ((3*H0_Chadwick15*low_f)/(np.pi*(E_bestH0/1e6)*R))**0.5

                    
                    legendText2 = 'bestH0 = {:.2f}nm'.format(bestH0)
                    plot_startH = np.concatenate((low_h, hPredict_bestH0[:]))
                    plot_startF = np.concatenate((low_f, fCompr[:NbPts_bestH0]))
                    thisAx2.plot(plot_startH[0], plot_startF[0], ls = '', 
                                  marker = 'o', color = 'skyblue', markersize = 5, label = legendText2)
                    thisAx2.plot(plot_startH, plot_startF, ls = '--', color = 'skyblue', linewidth = 1.2, zorder = 1)
                    thisAx2.legend(loc = 'upper right', prop={'size': 6})
                
                
                multiAxes = [thisAx2, thisAx3]
                
                for ax in multiAxes:
                    ax.title.set_text(titleText)
                    for item in ([ax.title, ax.xaxis.label, \
                                  ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(9)
                        
                        
                #### fig4 & fig5
                
                if not error_bestH0:
                    Npts_fitToPlot = dictRegionFit['Npts'][mask_selectFitsToPlot]
                    K_fitToPlot = dictRegionFit['K'][mask_selectFitsToPlot]
                    K_CIW_fitToPlot =dictRegionFit['K_CIW'][mask_selectFitsToPlot]
                    R2_fitToPlot = dictRegionFit['R2'][mask_selectFitsToPlot]
                    fitError_fitToPlot = dictRegionFit['fitError'][mask_selectFitsToPlot]
                    validatedFit_fitToPlot = dictRegionFit['validatedFit'][mask_selectFitsToPlot]
                    # dictColumnsRegionFit = {'regionFitNames' : '', 'K' : np.nan, 'K_CIW' : np.nan, 'R2' : np.nan,  
                    #                         'Npts' : np.nan, 'fitError' : True, 'validatedFit' : False} 
                    
                    plot_stressMasks_region = np.array(all_stressMasks_region)[mask_selectFitsToPlot]
                
                    # ax2[i] with the 1 line plot
                    if nRowsSubplot == 1:
                        thisAx4 = ax4[colSp]
                        thisAx5 = ax5[colSp]
                    elif nRowsSubplot >= 1:
                        thisAx4 = ax4[rowSp,colSp]
                        thisAx5 = ax5[rowSp,colSp]
                    
                    main_color = 'k' # colorList10[0]
                    
                    thisAx4.plot(hCompr,fCompr, color = main_color, marker = 'o', markersize = 2, ls = '', alpha = 0.8) # 'b-'
                    titleText = currentCellID + '__c' + str(i+1)
                    thisAx4.set_xlabel('h (nm)')
                    thisAx4.set_ylabel('f (pN)')
    
                    for k in range(len(fitsToPlot)):
                        fit = fitsToPlot[k]
                        stressMask_fit = plot_stressMasks_region[k]
                        fCompr_fit = fCompr[stressMask_fit]
                        hCompr_fit = hCompr[stressMask_fit]
                        
                        Npts_plot = len(fCompr_fit)
                        
                        if Npts_plot < 5:
                            fitError_plot = True
                        else:
                            # (E, H0, hPredict, R2, Chi2, confIntE, confIntH0, fitError)
                            resTuplePlot = compressionFitChadwick(hCompr_fit, fCompr_fit, DIAMETER)
                            hPredict_plot, fitError_plot = resTuplePlot[2], resTuplePlot[-1]
                        
                        color = gs.colorList30[k]
                        legendText4 = ''
                        if not fitError_plot:
                            if fit not in alreadyLabeled4:
                                alreadyLabeled4.append(fit)
                                legendText4 += '' + fit + ''
                                thisAx4.plot(hPredict_plot, fCompr_fit, color = color, ls = '-', linewidth = 1.8, 
                                             label = legendText4)
                            elif fit in alreadyLabeled4:
                                thisAx4.plot(hPredict_plot, fCompr_fit, color = color, ls = '-', linewidth = 1.8)
                        

                    
                    thisAx5.set_xlabel('Strain')
                    thisAx5.set_ylabel('Stress (Pa)')
                    if not error_bestH0:
                        thisAx5.plot(strainCompr, stressCompr, color = main_color, 
                                     marker = 'o', markersize = 2, ls = '', alpha = 0.8)
                        
                        for k in range(len(fitsToPlot)):
                            fit = fitsToPlot[k]
                            Npts_fit = Npts_fitToPlot[k]
                            K_fit = K_fitToPlot[k]
                            R2_fit = R2_fitToPlot[k]
                            fitError_fit = fitError_fitToPlot[k]
                            validatedFit_fit = validatedFit_fitToPlot[k]
                            stressMask_fit = plot_stressMasks_region[k]
                            
                            stressCompr_fit = stressCompr[stressMask_fit]
                            strainPredict_fit = list_strainPredict_fitToPlot[k]
                            
                            color = gs.colorList30[k]
                            legendText5 = ''
                            
                            if not fitError_fit:
                                if fit not in alreadyLabeled5:
                                    alreadyLabeled5.append(fit)
                                    legendText5 += '' + fit + ''
                                    thisAx5.plot(strainPredict_fit, stressCompr_fit,  
                                                 color = color, ls = '-', linewidth = 1.8, label = legendText5)
                                elif fit in alreadyLabeled5:
                                    thisAx5.plot(strainPredict_fit, stressCompr_fit,  
                                                 color = color, ls = '-', linewidth = 1.8)
                            else:
                                pass
    
                    multiAxes = [thisAx4, thisAx5]
                    
                    for ax in multiAxes:
                        ax.title.set_text(titleText)
                        for item in ([ax.title, ax.xaxis.label, \
                                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)
                        
                        
                #### fig6
                
                fitCentersPlot = fitCenters[mask_selectFitsToPlot]
                
                if nRowsSubplot == 1:
                    thisAx6 = ax6[colSp]
                elif nRowsSubplot >= 1:
                    thisAx6 = ax6[rowSp,colSp]
                
                
                thisAx6.set_xlabel('sigma (Pa)')
                thisAx6.set_xlim([0, 1200])
                thisAx6.set_ylabel('K (kPa)')
                
                
                if not fitError and not error_bestH0:
                    
                    relativeError = np.zeros(len(K_fitToPlot))
                    
                    for k in range(len(fitsToPlot)):
                        fit = fitsToPlot[k]
                        
                        K_fit = K_fitToPlot[k]
                        K_CIW_fit = K_CIW_fitToPlot[k]
                        fitError_fit = fitError_fitToPlot[k]
                        validatedFit_fit = validatedFit_fitToPlot[k]
                        stressMask_fit = plot_stressMasks_region[k]
                        
                        stressCompr_fit = stressCompr[stressMask_fit]
                        strainPredict_fit = list_strainPredict_fitToPlot[k]
    
                        color = gs.colorList30[k]
                        
                        if not fitError_fit:
                            
                            Err = K_CIW_fit
                            relativeError[k] = (Err/K_fit)
                            mec = None
                            thisAx6.errorbar([fitCentersPlot[k]], [K_fit/1000], yerr = [(Err/2)/1000],
                                          color = color, marker = 'o', ms = 5, mec = mec)                           
                            
                        
                    multiAxes = [thisAx6]
                    
                    for ax in multiAxes:
                        ax.title.set_text(titleText)
                        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + \
                                     ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)
                            
                
                        
                #### fig7

                if nRowsSubplot == 1:
                    thisAx7 = ax7[colSp]
                elif nRowsSubplot >= 1:
                    thisAx7 = ax7[rowSp,colSp]
                    
                thisAx7.set_xlabel('epsilon')
                thisAx7.set_ylabel('ratios')
                legendText7 = ''
                
                # def def2delta(e):
                #     d = 3*H0_Chadwick15*e
                #     return(d)
                
                # def delta2def(d):
                #     e = d/(3*H0_Chadwick15)
                #     return(e)
                
                # secax = thisAx7.secondary_xaxis('top', functions=(def2delta, delta2def))
                # secax.set_xlabel('delta (nm)')
                
                if not fitError and not error_bestH0:
                    
                    A = 2* ((deltaCompr*(DIAMETER/2000))**0.5)
                    largeX = A/(bestH0/1000)
                    smallX = deltaCompr/(DIAMETER/1000)
                    # legendText7 = 'H0_Chadwick15 = {:.2f}nm'.format(H0_Chadwick15)
                    
                    thisAx7.plot(strainCompr, largeX, color = 'red',     ls = '', marker = '+', label = 'a/H0',     markersize = 3)#, mec = 'k', mew = 0.5)
                    thisAx7.plot(strainCompr, smallX, color = 'skyblue', ls = '', marker = '+', label = 'delta/2R', markersize = 3)#, mec = 'k', mew = 0.5)
                    thisAx7.legend(loc = 'upper left', prop={'size': 5})
                    thisAx7.set_yscale('log')
                    thisAx7.set_ylim([5e-3,10])
                    minPlot, maxPlot = thisAx7.get_xlim()
                    thisAx7.set_xlim([0,maxPlot])
                    thisAx7.plot([0,maxPlot], [1, 1], color = 'k', ls = '--', lw = 0.5)
                    
                    # # thisAx7bis.tick_params(axis='y', labelcolor='b')
                    # thisAx7bis = thisAx7.twinx()
                    # color = 'firebrick'
                    # thisAx7bis.set_ylabel('F (pN)', color=color)
                    # thisAx7bis.plot(strainCompr, fCompr, color=color, lw = 0.5)
                    # thisAx7bis.set_ylim([0,1400])
                    # thisAx7bis.tick_params(axis='y', labelrotation = 50, labelsize = 10)
                    # # thisAx7bis.tick_params(axis='y', labelcolor=color)
                    # # thisAx7bis.set_yticks([0,500,1000,1500])
                    # # minh = np.min(tsDF['D3'].values-DIAMETER)
                    
                    
                    epsLim = (largeX < 1.0)
                    if len(strainCompr[epsLim]) > 0:
                        strainLimit = np.max(strainCompr[epsLim])
                        minPlot, maxPlot = thisAx5.get_ylim()
                        # thisAx5.plot([strainLimit, strainLimit], [minPlot, maxPlot], color = 'gold', ls = '--')
                        minPlot, maxPlot = thisAx7.get_ylim()
                        thisAx7.plot([strainLimit, strainLimit], [minPlot, maxPlot], color = 'gold', ls = '--')
                    
                            
                    multiAxes = [thisAx7] #, thisAx7bis]
                    
                    for ax in multiAxes:
                        ax.title.set_text(titleText + '\nbestH0 = {:.2f}nm'.format(bestH0))
                        for item in ([ax.title, ax.xaxis.label, \
                                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                            item.set_fontsize(9)


            #### (5) hysteresis (its definition may change)
            try:
                # results['hysteresis'].append(hCompr[0] - hRelax[-1])
                results['hysteresis'][i] = hCompr[0] - hRelax[-1]
                
            except:
                print('hysteresis computation failed, see hCompr and hRelax below:')
                print(hCompr, hRelax)
                print('Details of the delimitation of the curve')
                print(jStart, jMax, jStop, offsetStop, thisCompDf.shape[0])
                print(offsetStart2)
                print(thisCompDf.B.values)
                # results['hysteresis'].append(np.nan)
                
                
            #### Save stress-strain [2/3] 
            # Export a file "Timeseries_stress-strain"
            if not error_bestH0:
                if saveAnalysedTs:
                    ts_H0[iStart+jStart+offsetStart2:iStart+jMax+1].fill(bestH0)
                    ts_stress[iStart+jStart+offsetStart2:iStart+jMax+1] = stressCompr
                    ts_strain[iStart+jStart+offsetStart2:iStart+jMax+1] = strainCompr
                    
        
        #### (6) Deal with the non analysed compressions
        else: # The compression curve was detected as not suitable for analysis
            generalBug = True
            results['comments'][i] = 'Unspecified bug in the code'
            print(gs.ORANGE + currentCellID + ' - Compression no ' + str(i+1) + ' not suitable for analysis !' + gs.NORMAL)
            

    
    #### PLOT [3/4]
    
    if PLOT:
        #### fig1
        color = 'blue'
        ax1.plot(tsDF['T'].values, tsDF['D3'].values-DIAMETER, color = color, ls = '-', linewidth = 1, zorder = 1)
        (axm, axM) = ax1.get_ylim()
        ax1.set_ylim([min(0,axm), axM])
        if (max(tsDF['D3'].values-DIAMETER) > 200):
            ax1.set_yticks(np.arange(0, max(tsDF['D3'].values-DIAMETER), 100))
        
        twinAxis = True
        if twinAxis:
            ax1.tick_params(axis='y', labelcolor='b')
            ax1bis = ax1.twinx()
            color = 'firebrick'
            ax1bis.set_ylabel('F (pN)', color=color)
            ax1bis.plot(tsDF['T'].values, tsDF['F'].values, color=color)
            ax1bis.tick_params(axis='y', labelcolor=color)
            ax1bis.set_yticks([0,500,1000,1500])
            minh = np.min(tsDF['D3'].values-DIAMETER)
            ratio = min(1/abs(minh/axM), 5)
#             print(ratio)
            (axmbis, axMbis) = ax1bis.get_ylim()
            ax1bis.set_ylim([0, max(axMbis*ratio, 3*max(tsDF['F'].values))])

        
        
        #### fig3
        # Rescale fig3 axes
        eMin, eMax = 1, 0
        sMin, sMax = 1000, 0
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx3 = ax3[colSp]
            elif nRowsSubplot >= 1:
                thisAx3 = ax3[rowSp,colSp]
                
            if thisAx3.get_ylim()[1] > eMax:
                eMax = thisAx3.get_ylim()[1]
            if thisAx3.get_xlim()[1] > sMax:
                sMax = thisAx3.get_xlim()[1]
                
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx3 = ax3[colSp]
            elif nRowsSubplot >= 1:
                thisAx3 = ax3[rowSp,colSp]
                
            sMax = min(sMax, 2100)
            eMax = min(eMax, 0.5)
            
            thisAx3.set_xlim([0, sMax])
            thisAx3.set_ylim([0, eMax])
                
        #### fig4
        NL = len(alreadyLabeled4)
        titleVoid = ' '
        if NL > 16:
            titleVoid += '\n '
        fig4.suptitle(titleVoid)
        fig4.legend(loc='upper center', bbox_to_anchor=(0.5,1), ncol = min(8, NL))
                
        #### fig5
        
        NL = len(alreadyLabeled5)
        titleVoid = ' '
        if NL > 16:
            titleVoid += '\n '
        fig5.suptitle(titleVoid)
        fig5.legend(loc='upper center', bbox_to_anchor=(0.5,1), ncol = min(8, NL))
        
        # Rescale fig5 axes
        eMin, eMax = 1, 0
        sMin, sMax = 1000, 0
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx5 = ax5[colSp]
            elif nRowsSubplot >= 1:
                thisAx5 = ax5[rowSp,colSp]
                
            if thisAx5.get_xlim()[1] > eMax:
                eMax = thisAx5.get_xlim()[1]
            if thisAx5.get_ylim()[1] > sMax:
                sMax = thisAx5.get_ylim()[1]
                
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            # ax2[i-1] with the 1 line plot
            if nRowsSubplot == 1:
                thisAx5 = ax5[colSp]
            elif nRowsSubplot >= 1:
                thisAx5 = ax5[rowSp,colSp]
            
            sMax = min(sMax, 2100)
            eMax = min(eMax, 0.5)
            thisAx5.set_ylim([0, sMax])
            thisAx5.set_xlim([0, eMax])
                
        #### fig6
        # Rescale fig6 axes
        
        sMax = 0
        Kmax = 0

        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            if nRowsSubplot == 1:
                thisAx6 = ax6[colSp]
            elif nRowsSubplot >= 1:
                thisAx6 = ax6[rowSp,colSp]

            if thisAx6.get_xlim()[1] > sMax:
                sMax = thisAx6.get_xlim()[1]
            if thisAx6.get_ylim()[1] > Kmax:
                Kmax = thisAx6.get_xlim()[1]
        
        for i in range(Ncomp):
            colSp = i % nColsSubplot
            rowSp = i // nColsSubplot
            if nRowsSubplot == 1:
                thisAx6 = ax6[colSp]
            elif nRowsSubplot >= 1:
                thisAx6 = ax6[rowSp,colSp]
                
            Kmax = min(Kmax, 20)

            thisAx6.set_xlim([0, sMax])
            thisAx6.set_ylim([0, Kmax])
                
                
        Allfigs = [fig1,fig2,fig3,fig4,fig5,fig6,fig7]
        for fig in Allfigs:
            fig.tight_layout()

    
    #### PLOT [4/4]
    # Save the figures
    if PLOT:
        dpi = 150       
        figSubDir = 'MecaAnalysis_allCells_' + task
        ufun.archiveFig(fig1, name = currentCellID + '_01_h(t)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig2, name = currentCellID + '_02_F(h)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig3, name = currentCellID + '_03_sig(eps)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig4, name = currentCellID + '_04_F(h)_regionFits', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig5, name = currentCellID + '_05_sig(eps)_regionFits', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig6, name = currentCellID + '_06_K(s)', figSubDir = figSubDir, dpi = dpi)
        ufun.archiveFig(fig7, name = currentCellID + '_07_smallElements', figSubDir = figSubDir, dpi = dpi)
        plt.close('all')
    
    #### Save stress-strain [3/3] 
    # Export a file "Timeseries_stress-strain"
    if saveAnalysedTs:
        tsDF['H0'] = ts_H0
        tsDF['Stress'] = ts_stress
        tsDF['Strain'] = ts_strain
        saveName = f[:-4] + '_stress-strain.csv'
        savePath = os.path.join(cp.DirDataTimeseriesStressStrain, saveName)
        tsDF.to_csv(savePath, sep=';', index = False)
        if cp.CloudSaving != '':
            cloudSavePath = os.path.join(cp.DirCloudTimeseriesStressStrain, saveName)
            tsDF.to_csv(cloudSavePath, sep=';', index = False)
    
        
    #### (8) Convert to dataFrame before returning
    for k in results.keys():
        results[k] = np.array(results[k])
    result_df = pd.DataFrame(results)
    return(result_df)



def buildDf_meca(list_mecaFiles, dictColumnsMeca, task, expDf, PLOT=False):
    """
    Subfunction of computeGlobalTable_meca
    Create the dictionnary that will be converted in a pandas table in the end.
    """
    # expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    # Initialize the list of results dataFrames with an empty one, but with the right columns
    initDict = {k:[] for k in dictColumnsMeca.keys()}
    initDf = pd.DataFrame(initDict)
    list_resultDf = [initDf]
    
    Nfiles = len(list_mecaFiles)
    for f in list_mecaFiles: #[:10]:
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDF = pd.read_csv(tS_DataFilePath, sep = ';')
         # MAIN SUBFUNCTION
        current_resultDf = analyseTimeSeries_meca(f, current_tsDF, expDf, dictColumnsMeca, 
                                                  task, PLOT, saveAnalysedTs = True)
        list_resultDf.append(current_resultDf)
    mecaDf = pd.concat(list_resultDf)

    return(mecaDf)


def updateUiDf_meca(ui_fileSuffix, mecaDf):
    """
    """
    listColumnsUI = ['date','cellName','cellID','manipID','compNum',
                     'UI_Valid','UI_Comments']
    
    listDates = mecaDf['date'].unique()
    
    for date in listDates:
        ui_fileName = date + '_' + ui_fileSuffix
        
        try:
            savePath = os.path.join(cp.DirDataAnalysisUMS, (ui_fileName + '.csv'))
            uiDf = pd.read_csv(savePath, sep='\t')
            fromScratch = False
            print(gs.GREEN + date + ' : imported existing UI table' + gs.NORMAL)
            
        except:
            print(gs.DARKGREEN + date + ' : no existing UI table found' + gs.NORMAL)
            fromScratch = True
    
        new_uiDf = mecaDf[mecaDf['date'] == date][listColumnsUI[:5]]
        if not fromScratch:
            existingCellId = uiDf['cellID'].values
            new_uiDf = new_uiDf.loc[new_uiDf['cellID'].apply(lambda x : x not in existingCellId), :]
        
        nrows = new_uiDf.shape[0]
        new_uiDf['UI_Valid'] = np.ones(nrows, dtype = bool)
        new_uiDf['UI_Comments'] = np.array(['' for i in range(nrows)])
        
        if not fromScratch:
            new_uiDf = pd.concat([uiDf, new_uiDf], axis = 0, ignore_index=True)
            
        savePath = os.path.join(cp.DirDataAnalysisUMS, (ui_fileName + '.csv'))
        new_uiDf.sort_values(by=['cellID', 'compNum'], inplace = True)
        new_uiDf.to_csv(savePath, sep='\t', index = False)
        
        if cp.CloudSaving != '':
            CloudTimeSeriesFilePath = os.path.join(cp.DirCloudAnalysisUMS, (ui_fileName + '.csv'))
            new_uiDf.to_csv(CloudTimeSeriesFilePath, sep = ';', index=False)







def computeGlobalTable_meca(mode = 'fromScratch', task = 'all',
                            fileName = 'Global_MecaData', 
                            save = False, PLOT = False, \
                            source = 'Matlab', dictColumnsMeca=dictColumnsMeca):
    """
    Compute the GlobalTable_meca from the time series data files. \n
    > mode = 'fromScratch' or 'updateExisting':\n
    - 'fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.\n
    - 'updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new files.\n
    > task = 'all' or a string containing a file prefix, or several prefixes separated by ' & '.\n
    
    - 'all' will include in the analysis all the files that correspond to the chosen 'mode'.\n
    - giving file prefixes will include in the analysis all the files that start with this prefix and correspond to the chosen 'mode'. Examples: task = '22-05-03' or task = '22-05-03_M1 & 22-05-04_M2 & 22-05-05'.\n
    > fileName: the name of the table that will be created. If mode='updateExisting', it is this table that will be updated.\n
    > save: save the final table or not.\n
    > PLOT: save the plots or not.\n
    > source: 'Matlab' or 'Python', default is 'Python'.\n
    > dictColumnsMeca: dict that has to contain all the fields of the table that will be constructed AND their default values: dictColumnsMeca = {col1: defaultVal1, col2: defaultVal2, etc}.
    """
    top = time.time()
    
    ui_fileSuffix = 'UserManualSelection_MecaData'
    
    # 1. Initialization
    # 1.1 Get the experimental dataframe
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    
    # 1.2 Get the list of all meca files    
    suffixPython = '_PY'
    if source == 'Matlab':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('L40' in f)) and not (suffixPython in f))]
        
    elif source == 'Python':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('L40' in f)) and (suffixPython in f))]
    



    # 2. Get the existing table if necessary
    # 2.1
    if mode == 'fromScratch':
        existing_mecaDf = buildDf_meca([], dictColumnsMeca, 'fromScratch', expDf, PLOT=False)
    # 2.2
    elif mode == 'updateExisting':
        try:
            savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
            existing_mecaDf = pd.read_csv(savePath, sep=';')
        except:
            print('No existing table found')
    # 2.3
    else:
        existing_mecaDf = buildDf_meca([], dictColumnsMeca, 'fromScratch', expDf, PLOT=False)
    
    
    
    
    # 3. Select the files to analyse from the list according to the task
    list_taskMecaFiles = []
    # 3.1
    if task == 'all':
        list_taskMecaFiles = list_mecaFiles
    # 3.2
    else:
        task_list = task.split(' & ')
        for f in list_mecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            for t in task_list:
                if t in currentCellID:
                    list_taskMecaFiles.append(f)
                    break
                
    list_selectedMecaFiles = []
    # 3.3
    if mode == 'fromScratch':
        list_selectedMecaFiles = list_taskMecaFiles
        
    # 3.4
    elif mode == 'updateExisting':
        for f in list_taskMecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            if currentCellID not in existing_mecaDf.cellID.values:
                list_selectedMecaFiles.append(f)
                
                
                
        # 1.3 Exclude the ones that are not in expDf
        listExcluded = []
        listManips = expDf['manipID'].values
        for i in range(len(list_selectedMecaFiles)): 
            f = list_selectedMecaFiles[i]
            manipID = ufun.findInfosInFileName(f, 'manipID')
            if not manipID in listManips:
                listExcluded.append(f)
                
        for f in listExcluded:
            list_selectedMecaFiles.remove(f)
                
        if len(listExcluded) > 0:
            textExcluded = 'The following files were excluded from analysis\nbecause no matching experimental data was found:'
            print(gs.ORANGE + textExcluded)
            for f in listExcluded:
                print(f)
            print(gs.NORMAL)
            
                
                
    # 4. Run the analysis on the files, by blocks of 10
    Nfiles = len(list_selectedMecaFiles)
    mecaDf = existing_mecaDf
    for i in range(0, Nfiles, 10):
        i_start, i_stop = i, min(i+10, Nfiles)
        bloc_selectedMecaFiles = list_selectedMecaFiles[i_start:i_stop]
        # MAIN SUBFUNCTION
        new_mecaDf = buildDf_meca(bloc_selectedMecaFiles, dictColumnsMeca, task, expDf, PLOT)
        mecaDf = pd.concat([mecaDf, new_mecaDf])
        if save:
            saveName = fileName + '.csv'
            savePath = os.path.join(cp.DirDataAnalysis, saveName)
            mecaDf.to_csv(savePath, sep=';', index = False)
            if cp.CloudSaving != '':
                cloudSavePath = os.path.join(cp.DirCloudAnalysis, saveName)
                mecaDf.to_csv(cloudSavePath, sep=';', index = False)
            textSave = 'Intermediate save {:.0f}/{:.0f} successful !'.format((i+10)//10, ((Nfiles-1)//10)+1)
            print(gs.CYAN + textSave + gs.NORMAL)
         
            
    # 5. Final save
    if save:
        saveName = fileName + '.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        mecaDf.to_csv(savePath, sep=';', index = False)
        if cp.CloudSaving != '':
            cloudSavePath = os.path.join(cp.DirCloudAnalysis, saveName)
            mecaDf.to_csv(cloudSavePath, sep=';', index = False)
        print(gs.BLUE + 'Final save successful !' + gs.NORMAL)
    
    updateUiDf_meca(ui_fileSuffix, mecaDf)
    
    duration = time.time() - top
    print(gs.DARKGREEN + 'Total time: {:.0f}s'.format(duration) + gs.NORMAL)
    
    return(mecaDf)
            

    
def getGlobalTable_meca(fileName):
    try:
        savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
        mecaDf = pd.read_csv(savePath, sep=';')
        print('Extracted a table with ' + str(mecaDf.shape[0]) + ' lines and ' + str(mecaDf.shape[1]) + ' columns.')
    except:
        print('No existing table found')
        
    for c in mecaDf.columns:
        if 'Unnamed' in c:
            mecaDf = mecaDf.drop([c], axis=1)
        # if 'K_CIW_' in c:    
        #     mecaDf[c].apply(lambda x : x.strip('][').split(', ')).apply(lambda x : [float(x[0]), float(x[1])])
    
    if 'ExpDay' in mecaDf.columns:
        dateExemple = mecaDf.loc[mecaDf.index[0],'ExpDay']
        if not ('manipID' in mecaDf.columns):
            mecaDf['manipID'] = mecaDf['ExpDay'] + '_' + mecaDf['CellID'].apply(lambda x: x.split('_')[0])
            
    elif 'date' in mecaDf.columns:
        dateExemple = mecaDf.loc[mecaDf.index[0],'date']
        if re.match(ufun.dateFormatExcel, dateExemple):
            print('bad date')
        
    if not ('manipID' in mecaDf.columns):
        mecaDf['manipID'] = mecaDf['date'] + '_' + mecaDf['cellName'].apply(lambda x: x.split('_')[0])

        
    return(mecaDf)

# %%% (2.4) Fluorescence data

def getFluoData(save = False):
    # Getting the table
    fluoDataFile = 'FluoQuantification.csv'
    fluoDataFilePath = os.path.join(cp.DirDataAnalysis, fluoDataFile)
    fluoDF = pd.read_csv(fluoDataFilePath, sep=';',header=0)
    print('Extracted a table with ' + str(fluoDF.shape[0]) + ' lines and ' + str(fluoDF.shape[1]) + ' columns.')
    # Cleaning the table
    try:
        for c in fluoDF.columns:
            if 'Unnamed' in c:
                fluoDF = fluoDF.drop([c], axis=1)
        
    except:
        print('Unexpected bug with the cleaning step')

    if save:
        saveName = 'FluoQuantification.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        fluoDF.to_csv(savePath, sep=';')

    
    return(fluoDF)

# %%% (2.5) Oscillations

def analyseTimeSeries_sinus(f, tsDF, expDf, listColumns, PLOT, PLOT_SHOW):
    
    plotSmallElements = True
    
    #### (0) Import experimental infos
    split_f = f.split('_')
    tsDF.dx, tsDF.dy, tsDF.dz, tsDF.D2, tsDF.D3 = tsDF.dx*1000, tsDF.dy*1000, tsDF.dz*1000, tsDF.D2*1000, tsDF.D3*1000
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]

    # Deal with the asymmetric pair case : the diameter can be for instance 4503 (float) or '4503_2691' (string)
    diameters = thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'].split('_')
    if len(diameters) == 2:
        DIAMETER = (int(diameters[0]) + int(diameters[1]))/2.
    else:
        DIAMETER = int(diameters[0])
    
    EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])

    results = {}
    for c in listColumnsMeca:
        results[c] = []

    nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
    
    if 'sinus' in EXPTYPE:
        pass


def createDataDict_sinus(listFiles, listColumns, PLOT):
    """
    Subfunction of computeGlobalTable_meca
    Create the dictionnary that will be converted in a pandas table in the end.
    """
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    tableDict = {}
    Nfiles = len(listFiles)
    PLOT_SHOW = (Nfiles==1)
    PLOT_SHOW = 0
    if not PLOT_SHOW:
        plt.ioff()
    for c in listColumns:
        tableDict[c] = []
    for f in listFiles: #[:10]:
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDF = pd.read_csv(tS_DataFilePath, sep = ';')
         # MAIN SUBFUNCTION
        current_resultDict = analyseTimeSeries_sinus(f, current_tsDF, expDf, 
                                                    listColumns, PLOT, PLOT_SHOW)
        for k in current_resultDict.keys():
            tableDict[k] += current_resultDict[k]
#     for k in tableDict.keys():
#         print(k, len(tableDict[k]))
    return(tableDict)

# def computeGlobalTable_sinus(task = 'fromScratch', fileName = 'Global_Sinus', save = False, PLOT = False, \
#                             source = 'Matlab', listColumns=listColumnsMeca):
#     """
#     Compute the GlobalTable_Sinus from the time series data files.
#     Option task='fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.
#     Option task='updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new fils.
#     listColumns have to contain all the fields of the table that will be constructed.
#     """
#     top = time.time()
    
# #     list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
# #                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
# #                       and ('R40' in f))] # Change to allow different formats in the future
    
#     suffixPython = '_PY'
#     if source == 'Matlab':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and ('R40' in f) and not (suffixPython in f))]
        
#     elif source == 'Python':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and ('sin' in f) and (suffixPython in f))]
#         # print(list_mecaFiles)
    
# #     print(list_mecaFiles)
    
#     if task == 'fromScratch':
#         # create a dict containing the data
#         tableDict = createDataDict_sinus(list_mecaFiles, listColumns, PLOT) # MAIN SUBFUNCTION
#         # create the dataframe from it
#         DF = pd.DataFrame(tableDict)
        
#         # last step: now that the dataFrame is complete, one can use "compStartTimeThisDay" col to compute the start time of each compression relative to the first one done this day.
#         allDates = list(DF['date'].unique())
#         for d in allDates:
#             subDf = DF.loc[DF['date'] == d]
#             experimentStartTime = np.min(subDf['compStartTimeThisDay'])
#             DF['compStartTimeThisDay'].loc[DF['date'] == d] = DF['compStartTimeThisDay'] - experimentStartTime
        
#     elif task == 'updateExisting':
#         # get existing table
#         try:
#             savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
#             existing_mecaDf = pd.read_csv(savePath, sep=';')
#         except:
#             print('No existing table found')
            
#         # find which of the time series files are new
#         new_list_mecaFiles = []
#         for f in list_mecaFiles:
#             split_f = f.split('_')
#             currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
#             if currentCellID not in existing_mecaDf.cellID.values:
#                 new_list_mecaFiles.append(f)
                
#         # create the dict with new data
#         new_tableDict = createDataDict_meca(new_list_mecaFiles, listColumnsMeca, PLOT) # MAIN SUBFUNCTION
#         # create the dataframe from it
#         new_mecaDf = pd.DataFrame(new_tableDict)
#         # fuse the existing table with the new one
#         DF = pd.concat([existing_mecaDf, new_mecaDf])
        
#     else: # If task is neither 'fromScratch' nor 'updateExisting'
#     # Then task can be a substring that can be in some timeSeries file !
#     # It will create a table with only these files, WITHOUT SAVING IT !
#     # But it can plot figs from it.
#         # save = False
#         task_list = task.split(' & ')
#         new_list_mecaFiles = []
#         for f in list_mecaFiles:
#             split_f = f.split('_')
#             currentCellID = split_f[0] + '_' + split_f[1] + '_' + split_f[2] + '_' + split_f[3]
#             for t in task_list:
#                 if t in currentCellID:
#                     new_list_mecaFiles.append(f)
#                     break
#         # create the dict with new data
#         new_tableDict = createDataDict_meca(new_list_mecaFiles, listColumnsMeca, PLOT) # MAIN SUBFUNCTION
#         # create the dataframe from it
#         DF = pd.DataFrame(new_tableDict)
    
#     for c in DF.columns:
#             if 'Unnamed' in c:
#                 DF = DF.drop([c], axis=1)
    
#     if save:
#         saveName = fileName + '.csv'
#         savePath = os.path.join(cp.DirDataAnalysis, saveName)
#         DF.to_csv(savePath, sep=';')
    
#     delta = time.time() - top
#     print(delta)
    
#     return(DF)



# %% (5) General import functions
    
# %%% Main functions

def getAnalysisTable(fileName):
    if not fileName[-4:] == '.csv':
        ext = '.csv'
    else:
        ext = ''
        
    try:
        path = os.path.join(cp.DirDataAnalysis, (fileName + ext))
        df = pd.read_csv(path, sep=';')
        print(gs.CYAN + 'Analysis table has ' + str(df.shape[0]) + ' lines and ' + \
              str(df.shape[1]) + ' columns.' + gs.NORMAL)
    except:
        print(gs.BRIGHTRED + 'No analysis table found' + gs.NORMAL)
        return()
        
    for c in df.columns:
        if 'Unnamed' in c:
            df = df.drop([c], axis=1)
            
    if 'CellName' in df.columns and 'CellID' in df.columns:
        shortCellIdColumn = 'CellID'
    elif 'cellName' in df.columns and 'cellID' in df.columns:
        shortCellIdColumn = 'cellName'
    
    if 'ExpDay' in df.columns:
        dateColumn = 'ExpDay'
    elif 'date' in df.columns:
        dateColumn = 'date'
        
    # try:
    dateExemple = df.loc[df.index[0],dateColumn]
    df = ufun.correctExcelDatesInDf(df, dateColumn, dateExemple)
    # except:
    #     print(gs.ORANGE + 'Problem in date correction' + gs.NORMAL)
        
    try:
        if not ('manipID' in df.columns):
            df['manipID'] = df[dateColumn] + '_' + df[shortCellIdColumn].apply(lambda x: x.split('_')[0])
    except:
        print(gs.ORANGE + 'Could not infer Manip Ids' + gs.NORMAL)
        
    return(df)



def getMergedTable(fileName, DirDataExp = cp.DirRepoExp, suffix = cp.suffix,
                   mergeExpDf = True, mergeFluo = False, mergeUMS = False):
    
    df = getAnalysisTable(fileName)
    
    
    if mergeFluo:
        fluoDf = getFluoData()
        df = pd.merge(fluoDf, df, how="right", on='cellID', suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,
        )
    df = ufun.removeColumnsDuplicate(df)
        
    
    if mergeUMS:
        if 'ExpDay' in df.columns:
            dateColumn = 'ExpDay'
        elif 'date' in df.columns:
            dateColumn = 'date'
        listDates = df[dateColumn].unique()
        
        listFiles_UMS = os.listdir(cp.DirDataAnalysisUMS)
        listFiles_UMS_matching = []
        # listPaths_UMS = [os.path.join(DirDataAnalysisUMS, f) for f in listFiles_UMS]
        
        for f in listFiles_UMS:
            for d in listDates:
                if d in f:
                    listFiles_UMS_matching.append(f)
            
        listPaths_UMS_matching = [os.path.join(cp.DirDataAnalysisUMS, f) for f in listFiles_UMS_matching]
        listDF_UMS_matching = [pd.read_csv(p, sep = '\t') for p in listPaths_UMS_matching]
        
        umsDf = pd.concat(listDF_UMS_matching)
        # f_filterCol = lambda x : x not in ['date', 'cellName', 'manipID']
        # umsCols = umsDf.columns[np.array([f_filterCol(c) for c in umsDf.columns])]   
    
        df = pd.merge(umsDf, df, how="right", on=['cellID', 'compNum'], suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,[umsCols]
        )
        
    df = ufun.removeColumnsDuplicate(df)
    
    
    
    if mergeExpDf:
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = suffix)
        df = pd.merge(expDf, df, how="inner", on='manipID', suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,
        )
        
    df = ufun.removeColumnsDuplicate(df)
    
    print(gs.CYAN + 'Merged table has ' + str(df.shape[0]) + ' lines and ' \
          + str(df.shape[1]) + ' columns.' + gs.NORMAL)
        
    return(df)



# %% OLD CODE



#### Old version of computeGlobalTable_meca ; archived on the 22-09-03

# def computeGlobalTable_meca_v0(task = 'fromScratch', fileName = 'Global_MecaData', 
#                             save = False, PLOT = False, \
#                             source = 'Matlab', dictColumnsMeca=dictColumnsMeca,
#                             ui_fileSuffix = 'UserManualSelection_MecaData'):
#     """
#     Compute the GlobalTable_meca from the time series data files.
#     Option task='fromScratch' will analyse all the time series data files and construct a new GlobalTable from them regardless of the existing GlobalTable.
#     Option task='updateExisting' will open the existing GlobalTable and determine which of the time series data files are new ones, and will append the existing GlobalTable with the data analysed from those new fils.
#     DEPRECATED : listColumnsMeca have to contain all the fields of the table that will be constructed.
#     dictColumnsMeca have to contain all the fields of the table that will be constructed AND their default values !
#     """
#     top = time.time()
    
# #     list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
# #                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
# #                       and ('R40' in f))] # Change to allow different formats in the future
    
#     suffixPython = '_PY'
#     if source == 'Matlab':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and (('R40' in f) or ('L40' in f)) and not (suffixPython in f))]
        
#     elif source == 'Python':
#         list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                       if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
#                       and (('R40' in f) or ('L40' in f)) and (suffixPython in f))]

    
#     if task == 'fromScratch':
#         # create the dataFrame with new data
#         mecaDf = buildDf_meca(list_mecaFiles, dictColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
        
#         updateUiDf_meca(ui_fileSuffix, mecaDf)
        
#         # last step: now that the dataFrame is complete, one can use "compStartTimeThisDay" col to compute the start time of each compression relative to the first one done this day.
#         allDates = list(mecaDf['date'].unique())
#         for d in allDates:
#             subDf = mecaDf.loc[mecaDf['date'] == d]
#             experimentStartTime = np.min(subDf['compStartTimeThisDay'])
#             mecaDf['compStartTimeThisDay'].loc[mecaDf['date'] == d] = mecaDf['compStartTimeThisDay'] - experimentStartTime
        
#     elif task == 'updateExisting':
#         # get existing table
#         try:
#             savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
#             existing_mecaDf = pd.read_csv(savePath, sep=';')
#         except:
#             print('No existing table found')
            
#         # find which of the time series files are new
#         new_list_mecaFiles = []
#         for f in list_mecaFiles:
#             currentCellID = ufun.findInfosInFileName(f, 'cellID')
#             if currentCellID not in existing_mecaDf.cellID.values:
#                 new_list_mecaFiles.append(f)
                
#         # create the dataFrame with new data
#         new_mecaDf = buildDf_meca(new_list_mecaFiles, dictColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
#         # fuse the existing table with the new one
#         mecaDf = pd.concat([existing_mecaDf, new_mecaDf])
        
#         updateUiDf_meca(ui_fileSuffix, mecaDf)
        
#     else: # If task is neither 'fromScratch' nor 'updateExisting'
#     # Then task can be a substring that can be in some timeSeries file !
#     # It will create a table with only these files !

#         task_list = task.split(' & ')
#         new_list_mecaFiles = []
#         for f in list_mecaFiles:
#             currentCellID = ufun.findInfosInFileName(f, 'cellID')
#             for t in task_list:
#                 if t in currentCellID:
#                     new_list_mecaFiles.append(f)
#                     break
                
#         # create the dataFrame with new data
#         mecaDf = buildDf_meca(new_list_mecaFiles, dictColumnsMeca, task, PLOT) # MAIN SUBFUNCTION
#         updateUiDf_meca(ui_fileSuffix, mecaDf)
    
#     for c in mecaDf.columns:
#         if 'Unnamed' in c:
#             mecaDf = mecaDf.drop([c], axis=1)
    
#     if save:
#         saveName = fileName + '.csv'
#         savePath = os.path.join(cp.DirDataAnalysis, saveName)
#         mecaDf.to_csv(savePath, sep=';', index = False)
    
#     duration = time.time() - top
#     print(gs.DARKGREEN + 'Total time: {:.0f}s'.format(duration) + gs.NORMAL)
    
#     return(mecaDf)






#### Old version of getGlobalTable ; archived on the 22-08-26

# def getGlobalTable_v0(kind, DirDataExp = cp.DirRepoExp):
#     if kind == 'ctField':
#         GlobalTable = getGlobalTable_ctField()
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(expDf, GlobalTable, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", on='cellID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
#         # print(GlobalTable_ctField.head())
    
#     elif kind == 'ctField_py':
#         GlobalTable = getGlobalTable_ctField('Global_CtFieldData_Py')
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(expDf, GlobalTable, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", on='cellID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
#         # print(GlobalTable_ctField.head())
        
#         # return(GlobalTable_ctField)

#     elif kind == 'meca_matlab':
#         GlobalTable = getGlobalTable_meca('Global_MecaData')
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='CellName', right_on='cellID'
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
#         # print(GlobalTable.tail())
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         # return(GlobalTable_meca_Matlab)


#     elif kind == 'meca_py':
#         GlobalTable = getGlobalTable_meca('Global_MecaData_Py')
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID'
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
        
#         # print(GlobalTable_meca_Py.tail())
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         # return(GlobalTable_meca_Py)


#     elif kind == 'meca_py2':
#         GlobalTable = getGlobalTable_meca('Global_MecaData_Py2')
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

#         # print(GlobalTable_meca_Py2.tail())
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         # return(GlobalTable_meca_Py2)
    
#     elif kind == 'meca_nonLin':
#         GlobalTable = getGlobalTable_meca('Global_MecaData_NonLin_Py')
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

#         # print(GlobalTable_meca_nonLin.tail())
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         # return(GlobalTable_meca_nonLin)
    
#     elif kind == 'meca_MCA':
#         GlobalTable = getGlobalTable_meca('Global_MecaData_MCA')
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

#         # print(GlobalTable_meca_nonLin.tail())
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         # return(GlobalTable)
        
#     elif kind == 'meca_HoxB8':
#         GlobalTable = getGlobalTable_meca('Global_MecaData_HoxB8')
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         ui_fileName = 'UserManualSelection_MecaData'
#         # uiDf, success = get_uiDf(ui_fileName)
#         GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         # GlobalTable = pd.merge(GlobalTable, uiDf, 
#         #                        how="left", left_on=['cellID', 'compNum'], right_on=['cellID', 'compNum'],
#         # #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         # #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         # )
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')

#         # return(GlobalTable)
    
#     else:
#         GlobalTable = getGlobalTable_meca(kind)
#         expDf = ufun.getExperimentalConditions(DirDataExp, suffix = cp.suffix)
#         fluoDf = getFluoData()
#         GlobalTable = pd.merge(GlobalTable, expDf, how="inner", on='manipID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         GlobalTable = pd.merge(GlobalTable, fluoDf, how="left", left_on='cellID', right_on='cellID',
#         #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
#         #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
#         )
#         print('Merged table has ' + str(GlobalTable.shape[0]) + ' lines and ' + str(GlobalTable.shape[1]) + ' columns.')
    
#         # print(GlobalTable_meca_nonLin.tail())
#         GlobalTable = ufun.removeColumnsDuplicate(GlobalTable)
    
#     if 'substrate' in GlobalTable.columns:
#         vals_substrate = GlobalTable['substrate'].values
#         if 'diverse fibronectin discs' in vals_substrate:
#             try:
#                 cellIDs = GlobalTable[GlobalTable['substrate'] == 'diverse fibronectin discs']['cellID'].values
#                 listFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
#                               if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
#                 for Id in cellIDs:
#                     for f in listFiles:
#                         if Id == ufun.findInfosInFileName(f, 'cellID'):
#                             thisCellSubstrate = ufun.findInfosInFileName(f, 'substrate')
#                             thisCellSubstrate = dictSubstrates[thisCellSubstrate]
#                             if not thisCellSubstrate == '':
#                                 GlobalTable.loc[GlobalTable['cellID'] == Id, 'substrate'] = thisCellSubstrate
#                 print('Automatic determination of substrate type SUCCEDED !')
                
#             except:
#                 print('Automatic determination of substrate type FAILED !')
    
#     return(GlobalTable)
