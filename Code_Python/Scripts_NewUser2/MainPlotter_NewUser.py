# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:27:55 2022
@author: Joseph Vermeil

MainPlotter_##.py - Script to plot graphs.
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

#!/usr/bin/env python
# coding: utf-8

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
import time
import random
import warnings
import itertools
import matplotlib
import numbers

from copy import copy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

#### Local Imports

import CortexPaths as cp
sys.path.append(cp.DirRepoPython)
sys.path.append(cp.DirRepoPythonUser)


import GraphicStyles as gs
import UtilityFunctions as ufun
import TrackAnalyser as taka


#### Pandas
pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns')
pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


####  Matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})

#### Graphic options
gs.set_default_options_jv()

#### Bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
output_notebook()


# %% Reminders

# %%%  Stats
# 
# (A) How to compute confidence intervals of fitted parameters with (1-alpha) confidence:
# 
#     0) from scipy import stats
#     1) df = nb_pts - nb_parms ; se = diag(cov)**0.5
#     2) Student t coefficient : q = stat.t.ppf(1 - alpha / 2, df)
#     3) ConfInt = [params - q*se, params + q*se]

# ## 




# %% TimeSeries functions


# %%% List files
allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                          if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
print(allTimeSeriesDataFiles)


# %%% Get a time series

df = taka.getCellTimeSeriesData('22-02-09_M1_P1_C7')


# %%% Plot a time series

# taka.plotCellTimeSeriesData('21-02-10_M1_P1_C2')
taka.plotCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_1Hz')

# %%% Plot a time series

taka.plotCellTimeSeriesData('22-03-21_M3_P1_C1_sin5-3_2Hz')

# %%% Variation on the plotCellTimeSeriesData function

# cellID = '22-02-09_M1_P1_C2'
# fromPython = True

# X = 'T'
# Y = np.array(['B', 'F'])
# units = np.array([' (mT)', ' (pN)'])
# timeSeriesDataFrame = taka.getCellTimeSeriesData(cellID, fromPython)
# print(timeSeriesDataFrame.shape)
# if not timeSeriesDataFrame.size == 0:
# #         plt.tight_layout()
# #         fig.show() # figsize=(20,20)
#     axes = timeSeriesDataFrame.plot(x=X, y=Y, kind='line', ax=None, subplots=True, sharex=True, sharey=False, layout=None, \
#                     figsize=(4,6), use_index=True, title = cellID + ' - f(t)', grid=None, legend=False, style=None, logx=False, logy=False, \
#                     loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, \
#                     table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False)
#     plt.gcf().tight_layout()
#     for i in range(len(Y)):
#         axes[i].set_ylabel(Y[i] + units[i])
        
#     axes[0].set_yticks([0,2,4,6,8,10])
#     axes[0].grid(axis="y")
#     axes[1].set_yticks([0,10,20,30,40,50,100,150])
#     axes[1].grid(axis="y")
#     plt.gcf().show()
# else:
#     print('cell not found')
    
#### Nice display for B and F

# cellID = '22-01-12_M2_P1_C3'
# cellID = '22-01-12_M1_P1_C6'
cellID = '22-02-09_M1_P1_C9'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

defaultColorCycle = plt.rcParams['axes.prop_cycle']
customColorCycle = cycler(color=['purple', 'red'])
plt.rcParams['axes.prop_cycle'] = customColorCycle

if not tSDF.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
    axes = tSDF[tSDF['T']<=40].plot(x=X, y=Y, kind='line', 
                    subplots=True, sharex=True, sharey=False,
                    figsize=(4,4), use_index=True)
    
    plt.gcf().tight_layout()
    for i in range(len(Y)):
        axes[i].set_ylabel(Y[i] + units[i])
        
    axes[0].set_yticks([0,10,20,30,40,50])
    axes[0].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    axes[0].grid(axis="y")
    axes[0].legend().set_visible(False)
    axes[1].set_yticks([k for k in range(0,1400,200)])
    axes[1].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    axes[1].grid(axis="y")
    axes[1].legend().set_visible(False)
    axes[1].set_xticks([0,10,20,30,40])
    axes[1].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    plt.gcf().show()
else:
    print('cell not found')
plt.rcParams['axes.prop_cycle'] = defaultColorCycle


# %%% Variation on the plotCellTimeSeriesData -> Sinus

# cellID = '22-01-12_M2_P1_C3'
# cellID = '22-01-12_M1_P1_C6'
cellID = '22-03-21_M3_P1_C1_sin5-3_1Hz'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

# defaultColorCycle = plt.rcParams['axes.prop_cycle']
# customColorCycle = cycler(color=['green', 'red'])
# plt.rcParams['axes.prop_cycle'] = customColorCycle

fig, ax = plt.subplots(1,1, figsize = (10,5))

if not tSDF.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
    tsDFplot = tSDF[(tSDF['T']>=3.04) & (tSDF['T']<=6.05)]
    
    ax.plot(tsDFplot['T'], 1000*(tsDFplot['D3']-4.503), color = gs.colorList40[30], label = 'Thickness')
    ax.set_ylabel('Thickness (nm)', color = gs.colorList40[30])
    ax.set_ylim([0, 350])
    ax.set_xlabel('Time (s)')

    axR = ax.twinx()
    axR.plot(tsDFplot['T'], tsDFplot['F'], color = gs.colorList40[23], label = 'Force')
    axR.set_ylabel('Force (pN)', color = gs.colorList40[23])
    axR.set_ylim([0, 150])

    # axes = tSDF[(tSDF['T']>=20) & (tSDF['T']<=60)].plot(x=X, y=Y, kind='line', 
    #                 subplots=True, sharex=True, sharey=False,
    #                 figsize=(4,4), use_index=True)
    
    # plt.gcf().tight_layout()
    # for i in range(len(Y)):
    #     axes[i].set_ylabel(Y[i] + units[i])
        
    # axes[0].set_yticks([0,10,20,30,40,50])
    # axes[0].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[0].grid(axis="y")
    # axes[0].legend().set_visible(False)
    # axes[1].set_yticks([k for k in range(0,1400,200)])
    # axes[1].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[1].grid(axis="y")
    # axes[1].legend().set_visible(False)
    # axes[1].set_xticks([0,10,20,30,40])
    # axes[1].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    plt.gcf().show()
else:
    print('cell not found')
# plt.rcParams['axes.prop_cycle'] = defaultColorCycle

# %%% Variation on the plotCellTimeSeriesData -> Broken Ramp

# cellID = '22-01-12_M2_P1_C3'
# cellID = '22-01-12_M1_P1_C6'
cellID = '22-03-21_M4_P1_C3'
fromPython = True

X = 'T'
Y = np.array(['B', 'F'])
units = np.array([' (mT)', ' (pN)'])
tSDF = taka.getCellTimeSeriesData(cellID, fromPython)

# defaultColorCycle = plt.rcParams['axes.prop_cycle']
# customColorCycle = cycler(color=['green', 'red'])
# plt.rcParams['axes.prop_cycle'] = customColorCycle

fig, ax = plt.subplots(1,1, figsize = (6,3))

if not tSDF.size == 0:
#         plt.tight_layout()
#         fig.show() # figsize=(20,20)
    tsDFplot = tSDF[(tSDF['T']>=31) & (tSDF['T']<=34)]
    
    ax.plot(tsDFplot['T'], 1000*(tsDFplot['D3']-4.503), color = gs.colorList40[30], label = 'Thickness')
    ax.set_ylabel('Thickness (nm)', color = gs.colorList40[30])
    ax.set_ylim([0, 1200])
    ax.set_xlabel('Time (s)')

    axR = ax.twinx()
    axR.plot(tsDFplot['T'], tsDFplot['F'], color = gs.colorList40[23], label = 'Force')
    axR.set_ylabel('Force (pN)', color = gs.colorList40[23])
    axR.set_ylim([0, 1500])

    # axes = tSDF[(tSDF['T']>=20) & (tSDF['T']<=60)].plot(x=X, y=Y, kind='line', 
    #                 subplots=True, sharex=True, sharey=False,
    #                 figsize=(4,4), use_index=True)
    
    # plt.gcf().tight_layout()
    # for i in range(len(Y)):
    #     axes[i].set_ylabel(Y[i] + units[i])
        
    # axes[0].set_yticks([0,10,20,30,40,50])
    # axes[0].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[0].grid(axis="y")
    # axes[0].legend().set_visible(False)
    # axes[1].set_yticks([k for k in range(0,1400,200)])
    # axes[1].tick_params(axis='y', labelrotation = 0, labelsize = 10)
    # axes[1].grid(axis="y")
    # axes[1].legend().set_visible(False)
    # axes[1].set_xticks([0,10,20,30,40])
    # axes[1].tick_params(axis='x', labelrotation = 0, labelsize = 10)
    plt.gcf().show()
else:
    print('cell not found')
# plt.rcParams['axes.prop_cycle'] = defaultColorCycle

# %%% Plot multiple time series

allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
for f in allTimeSeriesDataFiles:
    if '22-02-09_M3' in f:
        taka.plotCellTimeSeriesData(f[:-4])

allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
for f in allTimeSeriesDataFiles:
    if '22-02-09_M2' in f:
        taka.plotCellTimeSeriesData(f[:-4])


# %%% Close all

plt.close('all')


# %%% Functions acting on the trajectories

listeTraj = taka.getCellTrajData('21-12-16_M1_P1_C10', Ntraj = 2)
listeTraj[1]['pos']

def testTraj(D0):
    trajDir = os.path.join(cp.DirDataTimeseries, 'Trajectories')
    allTimeSeriesDataFiles = [f for f in os.listdir(cp.DirDataTimeseries) 
                              if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) 
                                  and f.endswith(".csv"))]
    cellIDList = []
    for f in allTimeSeriesDataFiles:
        cellID = ufun.findInfosInFileName(f, 'cellID')
        if '21-12-08' in cellID:
            cellIDList.append(cellID)

    print(cellIDList)

    fig, ax = plt.subplots(1,2, figsize = (8,4))
    width = 15
    ax[0].set_title('Inside bead')
    ax[0].axis([-width,width,-width,width])
    ax[1].set_title('Outside bead')
    ax[1].axis([-width,width,-width,width])

    for C in (cellIDList):
        listeTraj = taka.getCellTrajData(C)
        iOut = (listeTraj[1]['pos'] == 'out')
        iIn = 1 - iOut
        dfIn, dfOut = listeTraj[iIn]['df'], listeTraj[iOut]['df']
        Xin = dfIn.X.values - dfIn.X.values[0]
        Yin = dfIn.Y.values - dfIn.Y.values[0]
        Xout = dfOut.X.values - dfOut.X.values[0]
        Yout = dfOut.Y.values - dfOut.Y.values[0]
        npts = len(Xin)
        for iii in range(1, npts):
            D2in = ((Xin[iii]-Xin[iii-1])**2 + (Yin[iii]-Yin[iii-1])**2) ** 0.5
            D2out = ((Xout[iii]-Xout[iii-1])**2 + (Yout[iii]-Yout[iii-1])**2) ** 0.5
            if D2in > D0 and D2out > D0:
                dxCorrIn = Xin[iii-1]-Xin[iii]
                dyCorrIn = Yin[iii-1]-Yin[iii]
                dxCorrOut = Xout[iii-1]-Xout[iii]
                dyCorrOut = Yout[iii-1]-Yout[iii]
                Xin[iii:] += dxCorrIn
                Yin[iii:] += dyCorrIn
                Xout[iii:] += dxCorrOut
                Yout[iii:] += dyCorrOut
        if np.max(np.abs(Xin)) < width and np.max(np.abs(Yin)) < width and np.max(np.abs(Xout)) < width and np.max(np.abs(Yout)) < width:
            ax[0].plot(Xin, Yin)
            ax[1].plot(Xout, Yout)

    plt.show()
    
testTraj(1)
    
testTraj(1.5)




# #############################################################################
# %% GlobalTables functions



# %%% Experimental conditions

expDf = ufun.getExperimentalConditions(cp.DirRepoExp, save=True , sep = ';')





# =============================================================================
# %%% Constant Field

# %%%% Update the table

# taka.computeGlobalTable_ctField(task='updateExisting', save=False)


# %%%% Refresh the whole table

# taka.computeGlobalTable_ctField(task = 'updateExisting', fileName = 'Global_CtFieldData_Py', save = True, source = 'Python') # task = 'updateExisting'


# %%%% Display

df = taka.getGlobalTable_ctField().head()






# =============================================================================
# %%% Mechanics

# %%%% Update the table

# taka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_Py', 
#                             save = False, PLOT = False, source = 'Matlab') # task = 'updateExisting'


# %%%% Refresh the whole table

# taka.computeGlobalTable_meca(task = 'updateExisting', fileName = 'Global_MecaData_Py2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% Drugs

drugTask = '22-03-30'
# taka.computeGlobalTable_meca(task = drugTask, fileName = 'Global_MecaData_Drugs_Py', 
#                             save = False, PLOT = True, source = 'Python') # task = 'updateExisting'


# %%%% Non-Lin

nonLinTask = '21-12-08 & 22-01-12 & 22-02-09'
# taka.computeGlobalTable_meca(task = nonLinTask, fileName = 'Global_MecaData_NonLin_Py', 
#                             save = False, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% MCA

MCAtask = '21-01-18 & 21-01-21'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA', 
#                             save = False, PLOT = False, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% HoxB8

HoxB8task = '22-05-03 & 22-05-04 & 22-05-05' #' & 22-05-04 & 22-05-05'
taka.computeGlobalTable_meca(task = HoxB8task, fileName = 'Global_MecaData_HoxB8', 
                            save = False, PLOT = False, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'

# %%%% Demo for Duya

Demo = '22-06-16' #' & 22-05-04 & 22-05-05'
# taka.computeGlobalTable_meca(task = Demo, fileName = 'Global_MecaData_Demo', 
#                             save = True, PLOT = True, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = MCAtask, fileName = 'Global_MecaData_MCA2', 
#                             save = True, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% Precise dates (to plot)

# taka.computeGlobalTable_meca(task = '22-02-09', fileName = 'Global_MecaData_Py2', save = False, PLOT = True, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = '21-01-18', fileName = 'Global_MecaData_Py2', save = False, PLOT = True, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = '21-01-21', fileName = 'Global_MecaData_Py2', save = False, PLOT = True, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = '22-02-09_M1', fileName = 'Global_MecaData_NonLin2_Py', 
#                             save = False, PLOT = True, source = 'Python') # task = 'updateExisting'
# taka.computeGlobalTable_meca(task = '21-01-18_M2_P1_C3', fileName = 'Global_MecaData_NonLin2_Py', 
#                             save = False, PLOT = True, source = 'Python') # task = 'updateExisting'
taka.computeGlobalTable_meca(task = '22-02-09_M1_P1_C3', fileName = 'aaa', 
                            save = False, PLOT = False, source = 'Python') # task = 'updateExisting'


# %%%% Display

df = taka.getGlobalTable_meca('Global_MecaData_Py2').tail()

# %%%% Test of Numi's CRAZY GOOD NEW STUFF :D

taka.computeGlobalTable_meca(task = '22-02-09_M1_P1_C3', fileName = 'aaa', 
                            save = True, PLOT = True, source = 'Python') # task = 'updateExisting'

# =============================================================================
# %%% Fluorescence

# %%%% Display

df = taka.getFluoData().head()




# #############################################################################
# %% > Data import & export

# %%% Examples

#### GlobalTable_ctField_Py
# GlobalTable_ctField_Py = taka.getMergedTable('Global_CtFieldData_Py')

#### GlobalTable_meca_Py2
# GlobalTable_meca_Py2 = taka.getMergedTable('Global_MecaData_Py2')

# GlobalTable_meca_MCA = taka.getGlobalTable(kind = 'meca_MCA')
# GlobalTable_meca_MCA = taka.getMergedTable('Global_MecaData_MCA2', mergeFluo = True)

#### Global_MecaData_HoxB8
# GlobalTable_meca_HoxB8 = taka.getMergedTable('Global_MecaData_HoxB8', mergeUMS = True)


# %% > Plotting Functions

# %%% Objects declaration

# Fill these according to your plots

renameDict1 = {'SurroundingThickness':'Thickness (nm) [b&a]',
               'surroundingThickness':'Thickness (nm) [b&a]',
               'ctFieldThickness':'Thickness at low force (nm)',
               'EChadwick': 'E Chadwick (Pa)',
               'medianThickness': 'Median Thickness (nm)',               
               'fluctuAmpli': 'Fluctuations Amplitude (nm)',               
               'meanFluoPeakAmplitude' : 'Fluo Intensity (a.u.)',               
               'none':'control',               
               'doxycyclin':'expressing iMC linker',               
               }

styleDict1 =  {'aSFL-A11 & none':{'color':gs.colorList40[20],'marker':'o'},              
                'aSFL-A11 & doxycyclin':{'color':gs.colorList40[30],'marker':'o'},
                'aSFL-F8 & none':{'color':gs.colorList40[21],'marker':'o'},              
                'aSFL-F8 & doxycyclin':{'color':gs.colorList40[31],'marker':'o'},
                'aSFL-E4 & none':{'color':gs.colorList40[22],'marker':'o'},              
                'aSFL-E4 & doxycyclin':{'color':gs.colorList40[32],'marker':'o'},             
               }




# %%% Main functions

# These functions use matplotlib.pyplot and seaborn libraries to display 1D categorical or 2D plots

def D1Plot(data, fig = None, ax = None, CondCol=[], Parameters=[], Filters=[], 
           Boxplot=True, AvgPerCell=False, cellID='cellID', co_order=[],
           stats=True, statMethod='Mann-Whitney', box_pairs=[], 
           figSizeFactor = 1, markersizeFactor=1, orientation = 'h', useHue = False, 
           stressBoxPlot = False, bypassLog = False, autoscale = True):
    
    data_filtered = data
    for fltr in Filters:
        data_filtered = data_filtered.loc[fltr]    
    NCond = len(CondCol)
    
    print(data_filtered.shape)
    
    if NCond == 1:
        CondCol = CondCol[0]
        
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
#         dictAggMean['EChadwick'] = 'median'
        data_filtered = group.agg(dictAggMean)
        
    data_filtered.sort_values(CondCol, axis=0, ascending=True, inplace=True)
    
    if len(co_order) > 0:
        p = getStyleLists_Sns(co_order, styleDict1)
    else:
        p = sns.color_palette()
    
    NPlots = len(Parameters)
    Conditions = list(data_filtered[CondCol].unique())
    if len(co_order) == 0:
        co_order = Conditions
    
    if fig == None:
        if orientation == 'h':
            fig, ax = plt.subplots(1, NPlots, figsize = (5*NPlots*NCond*figSizeFactor, 5))
        elif orientation == 'v':
            fig, ax = plt.subplots(NPlots, 1, figsize = (5*NCond*figSizeFactor, 5*NPlots))
    else:
        pass
    
    scaley = autoscale
    
    markersize = 5*markersizeFactor
    
    if NPlots == 1:
        ax = np.array([ax])
    
    for k in range(NPlots):

        if not bypassLog and (('EChadwick' in Parameters[k]) or ('KChadwick' in Parameters[k])):
            ax[k].set_yscale('log')


        if Boxplot:
            if stressBoxPlot:
                sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 2, 'alpha' : 0.8, 'zorder' : 4},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4},
                            capprops={"color": 'k', "linewidth": 2, 'alpha' : 0.7, 'zorder' : 4})
                            # scaley = scaley)
            else:
                sns.boxplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], 
                            width = 0.5, showfliers = False, order= co_order, 
                            medianprops={"color": 'darkred', "linewidth": 1.5, 'alpha' : 0.8, 'zorder' : 1},
                            boxprops={"facecolor": 'None', "edgecolor": 'k',"linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
        #                   boxprops={"color": color, "linewidth": 0.5},
                            whiskerprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1},
                            capprops={"color": 'k', "linewidth": 1, 'alpha' : 0.7, 'zorder' : 1})
                            # scaley = scaley)
            
            # data_filtered.boxplot(column=Parameters[k], by = CondCol, ax=ax[k],showfliers = False) # linewidth = 2, width = 0.5

        if stats:
            if len(box_pairs) == 0:
                box_pairs = makeBoxPairs(co_order)
            addStat_df(ax[k], data_filtered, box_pairs, Parameters[k], CondCol, test = statMethod)
            # add_stat_annotation(ax[k], x=CondCol, y=Parameters[k], data=data_filtered,box_pairs = box_pairs,test=statMethod, text_format='star',loc='inside', verbose=2)
        
        if not useHue:
            sns.swarmplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
                          size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p)
        else:
            sns.swarmplot(x=CondCol, y=Parameters[k], data=data_filtered, ax=ax[k], order = co_order,
                          size=markersize, edgecolor='k', linewidth = 1*markersizeFactor, palette = p, 
                          hue = 'manipID')
            legend = ax[k].legend()
            legend.remove()

        ax[k].set_xlabel('')
        ax[k].set_ylabel(Parameters[k])
        ax[k].tick_params(axis='x', labelrotation = 10)
        ax[k].yaxis.grid(True)           
    
    plt.rcParams['axes.prop_cycle'] = gs.my_default_color_cycle
    return(fig, ax)





def D2Plot(data, fig = None, ax = None, XCol='', YCol='', CondCol='', Filters=[], 
           cellID='cellID', AvgPerCell=False, xscale = 'lin', yscale = 'lin', 
           figSizeFactor = 1, markers = [], markersizeFactor = 1):
    data_filtered = data
    for fltr in Filters:
        data_filtered = data_filtered.loc[fltr]
    
    NCond = len(CondCol)    
    if NCond == 1:
        CondCol = CondCol[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
        data_filtered = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
        data_filtered.reset_index(level=0, inplace=True)
        
    Conditions = list(data_filtered[CondCol].unique())
    
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (8*figSizeFactor,5))
    else:
        pass
    
    markersize = 5 * markersizeFactor
    
    if xscale == 'log':
        ax.set_xscale('log')
    if yscale == 'log':
        ax.set_yscale('log')
    
    current_color_list = getStyleLists(Conditions, styleDict1).as_hex()
    cc = cycler(color=current_color_list)
    ax.set_prop_cycle(cc)
    
    im = 0
    for c in Conditions:
        Xraw = data_filtered[data_filtered[CondCol] == c][XCol].values
        Yraw = data_filtered[data_filtered[CondCol] == c][YCol].values
        XYraw = np.array([Xraw,Yraw]).T
        XY = XYraw[~np.isnan(XYraw).any(axis=1), :]
        X, Y = XY[:,0], XY[:,1]
        if len(markers) == 0:
            m = 'o'
        else:
            m = markers[im]
            im += 1
        
        if len(X) == 0:
            ax.plot([], [])
#             if modelFit:
#                 ax.plot([], [])
                
        elif len(X) > 0:                    
            ax.plot(X, Y, m, markersize = markersize, markeredgecolor='k', markeredgewidth = 1, label=c)
            
    ax.set_xlabel(XCol)
    ax.set_xlim([min(0,1.1*np.min(data_filtered[XCol])), 1.1*np.max(data_filtered[XCol])])
    ax.set_ylabel(YCol)
    if not yscale == 'log':
        ax.set_ylim([min(0,1.1*np.min(data_filtered[YCol])), 1.1*np.max(data_filtered[YCol])])
    ax.legend(loc='upper left')
    return(fig, ax)




def D2Plot_wFit(data, fig = None, ax = None, 
                XCol='', YCol='', CondCol='', 
                Filters=[], cellID='cellID', co_order = [],
                AvgPerCell=False, showManips = True,
                modelFit=False, modelType='y=ax+b',
                xscale = 'lin', yscale = 'lin', 
                figSizeFactor = 1, markersizeFactor = 1):
    
    data_filtered = data
    for fltr in Filters:
        data_filtered = data_filtered.loc[fltr]
    
    NCond = len(CondCol)    
    if NCond == 1:
        CondCol = CondCol[0]
    elif NCond > 1:
        newColName = ''
        for i in range(NCond):
            newColName += CondCol[i]
            newColName += ' & '
        newColName = newColName[:-3]
        data_filtered[newColName] = ''
        for i in range(NCond):
            data_filtered[newColName] += data_filtered[CondCol[i]].astype(str)
            data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x + ' & ')
        data_filtered[newColName] = data_filtered[newColName].apply(lambda x : x[:-3])
        CondCol = newColName
    
    if AvgPerCell:
        group = data_filtered.groupby(cellID)
        dictAggMean = getDictAggMean(data_filtered)
        data_filtered = group.agg(dictAggMean.pop(cellID)) #.reset_index(level=0, inplace=True)
        data_filtered.reset_index(level=0, inplace=True)
        
    Conditions = list(data_filtered[CondCol].unique())
    
    if len(co_order) > 0:
        try:
            gs.colorList, markerList = getStyleLists(co_order, styleDict1)
        except:
            gs.colorList, markerList = gs.colorList30, gs.markerList10
    else:
        co_order = Conditions
        gs.colorList, markerList = gs.colorList30, gs.markerList10
        
    gs.colorList = [gs.colorList40[32]]
    
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize = (8*figSizeFactor,5))
    else:
        pass
    
    markersize = 5 * markersizeFactor
    
    if xscale == 'log':
        ax.set_xscale('log')
    if yscale == 'log':
        ax.set_yscale('log')
    
#     current_color_list = getStyleLists(Conditions, styleDict1).as_hex()
#     cc = cycler(color=current_color_list)
#     ax.set_prop_cycle(cc)
    
#     if modelFit:
#         # Tweak the style cycle to plot for each condition: the points ('o') and then the fit ('-') with the same color.
# #         current_prop_cycle = plt.rcParams['axes.prop_cycle']
# #         current_color_list = prop_cycle.by_key()['color']
#         ncustom_color_list = list(np.array([current_color_list, current_color_list]).flatten(order='F'))
#         # if new_color_list was ['red', 'green', blue'], custom_color_list is now ['red', 'red', 'green', 'green', blue', blue']
#         cc = cycler(color=ncustom_color_list)
#         ax.set_prop_cycle(cc)
    
    for i in range(len(co_order)):
        c = co_order[i]
        color = gs.colorList[i]
#         marker = my_default_marker_list[i]
        Xraw = data_filtered[data_filtered[CondCol] == c][XCol].values
        Yraw = data_filtered[data_filtered[CondCol] == c][YCol].values
        Mraw = data_filtered[data_filtered[CondCol] == c]['manipID'].values
        XYraw = np.array([Xraw,Yraw]).T
        XY = XYraw[~np.isnan(XYraw).any(axis=1), :]
        X, Y = XY[:,0], XY[:,1]
        M = Mraw[~np.isnan(XYraw).any(axis=1)]
        if len(X) == 0:
            ax.plot([], [])
            if modelFit:
                ax.plot([], [])
                
        elif len(X) > 0:
            eqnText = ''

            if modelFit:
                print('Fitting condition ' + c + ' with model ' + modelType)
                if modelType == 'y=ax+b':
                    params, results = ufun.fitLine(X, Y) # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1f} X + {:.1f}".format(params[1], params[0])
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    print("Y = {:.5} X + {:.5}".format(params[1], params[0]))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    # fitY = params[1]*X + params[0]
                    # imin = np.argmin(X)
                    # imax = np.argmax(X)
                    # ax.plot([X[imin],X[imax]], [fitY[imin],fitY[imax]], '--', lw = '1',
                    #         color = color, zorder = 4)
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = params[1]*fitX + params[0]
                    ax.plot(fitX, fitY, '--', lw = '1', 
                            color = color, zorder = 4)

                elif modelType == 'y=A*exp(kx)':
                    params, results = ufun.fitLine(X, np.log(Y)) # Y=a*X+b ; params[0] = b,  params[1] = a
                    pval = results.pvalues[1] # pvalue on the param 'k'
                    eqnText += " ; Y = {:.1f}*exp({:.1f}*X)".format(params[0], params[1])
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    print("Y = {:.5}*exp({:.5}*X)".format(np.exp(params[0]), params[1]))
                    print("p-value on the 'k' coefficient: {:.4e}".format(pval))
                    # fitY = np.exp(params[0])*np.exp(params[1]*X)
                    # imin = np.argmin(X)
                    # imax = np.argmax(X)
                    # ax.plot([X[imin],X[imax]], [fitY[imin],fitY[imax]], '--', lw = '1',
                    #         color = color, zorder = 4)
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = np.exp(params[0])*np.exp(params[1]*fitX)
                    ax.plot(fitX, fitY, '--', lw = '1', 
                            color = color, zorder = 4)
                    
                elif modelType == 'y=k*x^a':
                    posValues = ((X > 0) & (Y > 0))
                    X, Y = X[posValues], Y[posValues]
                    params, results = ufun.fitLine(np.log(X), np.log(Y)) # Y=a*X+b ; params[0] = b,  params[1] = a
                    k = np.exp(params[0])
                    a = params[1]
                    R2 = results.rsquared
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    eqnText += " ; Y = {:.1e} * X^{:.1f}".format(k, a)
                    eqnText += " ; p-val = {:.3f}".format(pval)
                    eqnText += " ; R2 = {:.2f}".format(R2)
                    print("Y = {:.4e} * X^{:.4f}".format(k, a))
                    print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    print("R2 of the fit: {:.4f}".format(R2))
                    # fitY = k * X**a
                    # imin = np.argmin(X)
                    # imax = np.argmax(X)
                    # ax.plot([X[imin],X[imax]], [fitY[imin],fitY[imax]], '--', lw = '1', 
                    #         color = color, zorder = 4)
                    fitX = np.linspace(np.min(X), np.max(X), 100)
                    fitY = k * fitX**a
                    ax.plot(fitX, fitY, '--', lw = '1', 
                            color = color, zorder = 4)
                
                print('Number of values : {:.0f}'.format(len(Y)))
                print('\n')
            
#                 if showManips:
#                     allManipID = list(data_filtered[data_filtered[CondCol] == c]['manipID'].unique())
#                     dictMarker = {}
#                     markers = []
#                     for mi in range(len(allManipID)):
#                         dictMarker[allManipID[mi]] = my_default_marker_list[mi]
#                     for k in range(len(M)):
#                         thisManipID = M[k]
#                         markers = dictMarker[thisManipID]
#                 else:
#                     markers = ['o' for i in range(len(M))]
#                 markers = np.array(markers)
                
            ax.plot(X, Y, 
                    color = color, ls = '',
                    marker = 'o', markersize = markersize, markeredgecolor='k', markeredgewidth = 1, 
                    label = c + eqnText)
            
            
    ax.set_xlabel(XCol)
    ax.set_xlim([0.9*np.min(data_filtered[XCol]), 1.1*np.max(data_filtered[XCol])])
    ax.set_ylabel(YCol)
    if not yscale == 'log':
        ax.set_ylim([0.9*np.min(data_filtered[YCol]), 1.1*np.max(data_filtered[YCol])])
    ax.legend(loc='upper left')
    
    
    return(fig, ax)



# %%% Subfunctions


    
def getDictAggMean(df):
    dictAggMean = {}
    for c in df.columns:
        # print(c)
        S = df[c].dropna()
        lenNotNan = S.size
        if lenNotNan == 0:
            dictAggMean[c] = 'first'
        else:
            S.infer_objects()
            if S.dtype == bool:
                dictAggMean[c] = np.nanmin
            else:
                # print(c)
                # print(pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))))
                if pd.Series.all(S.apply(lambda x : isinstance(x, numbers.Number))):
                    dictAggMean[c] = np.nanmean
                else:
                    dictAggMean[c] = 'first'
                    
    if 'compNum' in dictAggMean.keys():
        dictAggMean['compNum'] = np.nanmax
                    
    return(dictAggMean)



# def getDictAggMean_V1(df):
#     dictAggMean = {}
#     for c in df.columns:
#     #         t = df[c].dtype
#     #         print(c, t)
#             try:
#                 if np.array_equal(df[c], df[c].astype(bool)):
#                     dictAggMean[c] = 'min'
#                 else:
#                     try:
#                         if not c.isnull().all():
#                             dictAggMean[c] = 'nanmean'
#                     except:
#                         dictAggMean[c] = 'first'
#             except:
#                     dictAggMean[c] = 'first'
#     return(dictAggMean)


def getAggDf(df, cellID, CondCol, Variables):
    
    allCols = df.columns.values
    group = df.groupby(cellID)
    
    dictAggCategories = {}
    Categories = ['date', 'cellName', 'manipID', 'experimentType', 'drug', 
                  'substrate', 'cell type', 'cell subtype', 'bead type', 'bead diameter']
    if CondCol not in Categories:
        Categories += [CondCol]
    for c in Categories:
        if c in allCols:
            dictAggCategories[c] = 'first'
    dfCats = group.agg(dictAggCategories)
    
    dictAggCount = {'compNum' : 'count'}
    renameDict = {'compNum' : 'Ncomp'}
    dfCount = group.agg(dictAggCount).rename(columns=renameDict)
    
    dictAggMeans = {}
    renameDict = {}
    for v in Variables:
        dictAggMeans[v] = 'mean'
        renameDict[v] = v + '_mean'
    dfMeans = group.agg(dictAggMeans).rename(columns=renameDict)
        
    dictAggStds = {}
    renameDict = {}
    for v in Variables:
        dictAggStds[v] = 'std'
        renameDict[v] = v + '_std'
    dfStds = group.agg(dictAggStds).rename(columns=renameDict)
    
    if 'ctFieldThickness' in Variables:
        dfStds['ctFieldThickness_std'] = group.agg({'ctFieldFluctuAmpli' : 'median'}).values
        #ctFieldThickness ; ctFieldFluctuAmpli
    
#     for v in Variables:
        
    
    dfAgg = pd.merge(dfCats, dfCount, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
    
    dfAgg = pd.merge(dfAgg, dfMeans, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
    
    dfAgg = pd.merge(dfAgg, dfStds, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
    
    return(dfAgg)


def getAggDf_weightedAvg(df, cellID, CondCol, Variables, WeightCols):
    
    def w_std(x, w):
        m = np.average(x, weights=w)
        v = np.average((x-m)**2, weights=w)
        std = v**0.5
        return(std)
    
    def nan2zero(x):
        if np.isnan(x):
            return(0)
        else:
            return(x)
    
    dictWeights = {}
    for i in range(len(Variables)):
        v = Variables[i]
        w = WeightCols[i]
        dictWeights[v] = w

    for v in Variables:
        df[v] = df[v].apply(nan2zero)
        df[dictWeights[v]] = df[dictWeights[v]].apply(nan2zero)
    cellIDvals = df['cellID'].unique()
    for v in Variables:
        for c in cellIDvals:
            if np.sum(df.loc[df['cellID'] == c, v].values) == 0:
                df.loc[df['cellID'] == c, dictWeights[v]].apply(lambda x: 0)
    
    
    
    allCols = df.columns.values
    group = df.groupby(cellID)
    
    dictAggCategories = {}
    Categories = ['date', 'cellName', 'manipID', 'experimentType', 'drug', 
                  'substrate', 'cell type', 'cell subtype', 'bead type', 'bead diameter']
    if CondCol not in Categories:
        Categories += [CondCol]
    for c in Categories:
        if c in allCols:
            dictAggCategories[c] = 'first'
    dfCats = group.agg(dictAggCategories)
    
    dictAggCount = {'compNum' : 'count'}
    renameDict = {'compNum' : 'Ncomp'}
    dfCount = group.agg(dictAggCount).rename(columns=renameDict) 
    
    dfAgg = pd.merge(dfCats, dfCount, how="inner", on='cellID',
    #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
    #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
    )
        
    dictAggMeans = {}
    renameDict = {}
    for i in range(len(Variables)):
        v = Variables[i]
        w = WeightCols[i]
        df_i = df[[cellID, v, w]]
        group_i = df_i.groupby(cellID)

        def WMfun(x):
            try:
                return(np.average(x, weights=df.loc[x.index, dictWeights[v]]))
            except:
                return(np.nan)
            
        def WSfun(x):
            try:
                return(w_std(x, df.loc[x.index, dictWeights[v]].values))
            except:
                return(np.nan)
        
        def Countfun(x):
            try:
                return(len(x.values[x.values != 0]))
            except:
                return(0)
        
        dictAggMeans[v] = WMfun
        renameDict[v] = v + '_Wmean'
        dfWMeans_i = group_i.agg({v : WMfun}).rename(columns={v : v + '_Wmean'})
        dfWStd_i = group_i.agg({v : WSfun}).rename(columns={v : v + '_Wstd'})
        dfWCount_i = group_i.agg({v : Countfun}).rename(columns={v : v + '_Count'})
        dfAgg = pd.merge(dfAgg, dfWMeans_i, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        dfAgg = pd.merge(dfAgg, dfWStd_i, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )
        
        dfAgg = pd.merge(dfAgg, dfWCount_i, how="inner", on='cellID',
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     suffixes=("_x", "_y"),copy=True,indicator=False,validate=None,
        )

    return(dfAgg)


def makeOrder(*args):
    order = []
    listeTuple = list(itertools.product(*args, repeat=1))
    for tup in listeTuple:
        tmpText = ''
        for word in tup:
            tmpText += word
            tmpText += ' & '
        tmpText = tmpText[:-3]
        order.append(tmpText)
    return(order)


def makeBoxPairs(O):
    return(list(itertools.combinations(O, 2)))


def renameAxes(axes, rD):
    try:
        N = len(axes)
    except:
        axes = [axes]
        N = 1
    for i in range(N):
        # set xticks
        xticksTextObject = axes[i].get_xticklabels()
        xticksList = [xticksTextObject[j].get_text() for j in range(len(xticksTextObject))]
        test_hasXLabels = (len(''.join(xticksList)) > 0)
        if test_hasXLabels:
            newXticksList = [rD.get(k, k) for k in xticksList]
            axes[i].set_xticklabels(newXticksList)
        
        # set xlabel
        xlabel = axes[i].get_xlabel()
        newXlabel = rD.get(xlabel, xlabel)
        axes[i].set_xlabel(newXlabel)
        # set ylabel
        ylabel = axes[i].get_ylabel()
        newYlabel = rD.get(ylabel, ylabel)
        axes[i].set_ylabel(newYlabel)
        

def addStat_df(ax, data, box_pairs, param, cond, test = 'Mann-Whitney', percentHeight = 98):
    refHeight = np.percentile(data[param].values, percentHeight)
    currentHeight = refHeight
    scale = ax.get_yscale()
    xTicks = ax.get_xticklabels()
    dictXTicks = {xTicks[i].get_text() : xTicks[i].get_position()[0] for i in range(len(xTicks))}
    for bp in box_pairs:
        c1 = data[data[cond] == bp[0]][param].values
        c2 = data[data[cond] == bp[1]][param].values
        if test=='Mann-Whitney':
            statistic, pval = st.mannwhitneyu(c1,c2)
        elif test=='Wilcox_2s':
            statistic, pval = st.wilcoxon(c1,c2, alternative = 'two-sided')
        elif test=='Wilcox_greater':
            statistic, pval = st.wilcoxon(c1,c2, alternative = 'greater')
        elif test=='Wilcox_less':
            statistic, pval = st.wilcoxon(c1,c2, alternative = 'less')
        elif test=='t-test':
            statistic, pval = st.ttest_ind(c1,c2)
        text = 'ns'
        if pval < 0.05 and pval > 0.01:
            text = '*'
        elif pval < 0.01 and pval > 0.001:
            text = '**'
        elif pval < 0.001 and pval < 0.001:
            text = '***'
        elif pval < 0.0001:
            text = '****'
        ax.plot([bp[0], bp[1]], [currentHeight, currentHeight], 'k-', lw = 1.5, zorder = 4)
        XposText = (dictXTicks[bp[0]]+dictXTicks[bp[1]])/2
        if scale == 'log':
            power = 0.01* (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight*(refHeight**power)
        else:
            factor = 0.02 * (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight + factor*refHeight
        ax.text(XposText, YposText, text, ha = 'center', color = 'k', size = 11, zorder = 4)
#         if text=='ns':
#             ax.text(posText, currentHeight + 0.025*refHeight, text, ha = 'center')
#         else:
#             ax.text(posText, currentHeight, text, ha = 'center')
        if scale == 'log':
            currentHeight = currentHeight*(refHeight**0.05)
        else:
            currentHeight =  currentHeight + 0.1*refHeight
    # ax.set_ylim([ax.get_ylim()[0], currentHeight])
    

def addStat_arrays(ax, X, Y, box_pairs, test = 'Mann-Whitney', percentHeight = 98):
    refHeight = np.percentile(Y, percentHeight)
    currentHeight = refHeight
    scale = ax.get_yscale()
    
    for bp in box_pairs:
        y1 = Y[bp[0], :]
        y2 = Y[bp[1], :]
        
        if test=='Mann-Whitney':
            statistic, pval = st.mannwhitneyu(y1, y2)
        elif test=='Wilcox_2s':
            statistic, pval = st.wilcoxon(y1, y2, alternative = 'two-sided')
        elif test=='Wilcox_greater':
            statistic, pval = st.wilcoxon(y1, y2, alternative = 'greater')
        elif test=='Wilcox_less':
            statistic, pval = st.wilcoxon(y1, y2, alternative = 'less')
        elif test=='t-test':
            statistic, pval = st.ttest_ind(y1, y2)
            
        text = 'ns'
        if pval < 0.05 and pval > 0.01:
            text = '*'
        elif pval < 0.01 and pval > 0.001:
            text = '**'
        elif pval < 0.001 and pval < 0.001:
            text = '***'
        elif pval < 0.0001:
            text = '****'
            
        ax.plot([X[bp[0]], X[bp[1]]], [currentHeight, currentHeight], 'k-', lw = 1)
        XposText = (X[bp[0]]+X[bp[1]])/2
        if scale == 'log':
            power = 0.01* (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight*(refHeight**power)
        else:
            factor = 0.03 * (text=='ns') + 0.000 * (text!='ns')
            YposText = currentHeight + factor*refHeight
        ax.text(XposText, YposText, text, ha = 'center', color = 'k')
#         if text=='ns':
#             ax.text(posText, currentHeight + 0.025*refHeight, text, ha = 'center')
#         else:
#             ax.text(posText, currentHeight, text, ha = 'center')
        if scale == 'log':
            currentHeight = currentHeight*(refHeight**0.05)
        else:
            currentHeight =  currentHeight + 0.15*refHeight
    ax.set_ylim([ax.get_ylim()[0], currentHeight])


def getStyleCycle(co_order, styleDict):
    colors = []
    markers = []
    linestyles = []
    linewidth = []

    for co in co_order:
        coStyle = styleDict[co]
        if 'color' in coStyle.keys():
            colors.append(coStyle['color'])
        else:
            colors.append('')
        if 'marker' in coStyle.keys():
            markers.append(coStyle['marker'])
        else:
            markers.append('')
        if 'linestyle' in coStyle.keys():
            linestyles.append(coStyle['marker'])
        else:
            linestyles.append('')
        if 'linewidth' in coStyle.keys():
            linewidth.append(coStyle['linewidth'])
        else:
            linewidth.append(1)
            
    cc = (cycler(color=colors) +
          cycler(linestyle=linestyles) +
          cycler(marker=markers) +
          cycler(linewidth=linewidth))
            
    return(cc)


def getStyleLists_Sns(co_order, styleDict):
    colors = []
    markers = []
    try:
        for co in co_order:
            coStyle = styleDict[co]
            if 'color' in coStyle.keys():
                colors.append(coStyle['color'])
            else:
                colors.append('')
            if 'marker' in coStyle.keys():
                markers.append(coStyle['marker'])
            else:
                markers.append('')
        palette = sns.color_palette(colors)
    except:
        palette = sns.color_palette()
    return(palette)

def getStyleLists(co_order, styleDict):
    colors = []
    markers = []
    for co in co_order:
        coStyle = styleDict[co]
        if 'color' in coStyle.keys():
            colors.append(coStyle['color'])
        else:
            colors.append('')
        if 'marker' in coStyle.keys():
            markers.append(coStyle['marker'])
        else:
            markers.append('')
    print(colors, markers)
    return(colors, markers)


def buildStyleDictMCA():
    # TBC
    styleDict = {}
    return(styleDict)



# %% Plots

# %%% Run here scripts to make plots





