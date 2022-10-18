# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:21:02 2022
@authors: Joseph Vermeil, Anumita Jawahar

UtilityFunctions.py - contains all kind of small functions used by CortExplore programs, 
to be imported with "import UtilityFunctions as ufun" and call with "cp.my_function".
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

# 1. Imports
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import statsmodels.api as sm

import os
import re
import time
import shutil
import numbers
import pyautogui
import matplotlib
import traceback
# import cv2

# import scipy
from scipy import interpolate
from scipy import signal

# import skimage
from skimage import io, filters, exposure, measure, transform, util, color
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec
from datetime import date, datetime

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import GlobalConstants as gc

# 2. Pandas settings
pd.set_option('mode.chained_assignment', None)

# 3. Plot settings
gs.set_default_options_jv()

# 4. Other settings
# These regex are used to correct the stupid date conversions done by Excel
# When given a date yy-mm-dd, excel will convert in yy/mm/20dd or yy/mm/19dd if yy >= 30
dateFormatExcel = re.compile(r'[1-2]\d{1}/\d{2}/(?:19|20)\d{2}') # matches X#/##/YY## where X in {1, 2} and YY in {19, 20}
dateFormatExcel2 = re.compile(r'[1-2]\d-\d{2}-(?:19|20)\d{2}') # matches X#-##-YY## where X in {1, 2} and YY in {19, 20}
dateFormatOk = re.compile(r'\d{2}-\d{2}-\d{2}') # Correct format yy-mm-dd we use



# %% (1) Utility functions

# %%% Data management

def getExperimentalConditions(DirExp = cp.DirRepoExp, save = False, suffix = cp.suffix):
    """
    Import the table with all the conditions in a clean way.
    It is a tedious function to read because it's doing a boring job:
    Converting strings into numbers when possible; 
    Converting commas into dots to correct for the French decimal notation; 
    Converting semicolon separated values into lists when needed; 
    \n
    NEW FEATURE: Thanks to "engine='python'" in pd.read_csv() the separator can now be detected automatically !
    """
    
    #### 0. Import the table
    if suffix == '':
        experimentalDataFile = 'ExperimentalConditions.csv'
    else:
        experimentalDataFile = 'ExperimentalConditions' + suffix + '.csv'
        
    experimentalDataFilePath = os.path.join(DirExp, experimentalDataFile)
    expDf = pd.read_csv(experimentalDataFilePath, sep=None, header=0, engine='python')
    # print(gs.BLUE + 'Importing Experimental Conditions' + gs.NORMAL)
    print(gs.BLUE + 'Experimental Conditions Table has ' + str(expDf.shape[0]) + ' lines and ' + str(expDf.shape[1]) + ' columns' + gs.NORMAL)
    #### 1. Clean the table
    
    #### 1.1 Remove useless columns
    for c in expDf.columns:
        if 'Unnamed' in c:
            expDf = expDf.drop([c], axis=1)
        if '.1' in c:
            expDf = expDf.drop([c], axis=1)
    expDf = expDf.convert_dtypes()

    #### 1.2 Convert commas into dots
    listTextColumns = []
    for col in expDf.columns:
        try:
            if expDf[col].dtype == 'string':
                listTextColumns.append(col)
        except:
            pass
    expDf[listTextColumns] = expDf[listTextColumns].apply(lambda x: x.str.replace(',','.'))

    #### 1.3 Format 'scale'
    expDf['scale pixel per um'] = expDf['scale pixel per um'].astype(float)
    
    #### 1.4 Format 'optical index correction'
    try: # In case the format is 'n1/n2'
        expDf['optical index correction'] = \
                  expDf['optical index correction'].apply(lambda x: x.split('/')[0]).astype(float) \
                / expDf['optical index correction'].apply(lambda x: x.split('/')[1]).astype(float)
        print(gs.ORANGE + 'optical index correction : format changed' + gs.NORMAL)
    except:
        pass
    
    #### 1.5 Format 'magnetic field correction'
    expDf['magnetic field correction'] = expDf['magnetic field correction'].astype(float)
    
    #### 1.6 Format 'with fluo images'
    expDf['with fluo images'] = expDf['with fluo images'].astype(bool)

    # #### 1.7 Format 'ramp field'
    # try:
    #     print(ORANGE + 'ramp field : converted to list successfully' + gs.NORMAL)
    #     expDf['ramp field'] = \
    #     expDf['ramp field'].apply(lambda x: [x.split(';')[0], x.split(';')[1]] if not pd.isnull(x) else [])
    # except:
    #     pass

    #### 1.8 Format 'date'
    dateExemple = expDf.loc[expDf.index[1],'date']
    if re.match(dateFormatExcel, dateExemple):
        print(gs.ORANGE + 'dates : format corrected' + gs.NORMAL)
        expDf.loc[:,'date'] = expDf.loc[:,'date'].apply(lambda x: x.split('/')[0] + '-' + x.split('/')[1] + '-' + x.split('/')[2][2:])        
    elif re.match(dateFormatExcel2, dateExemple):
        print(gs.ORANGE + 'dates : format corrected' + gs.NORMAL)
        expDf.loc[:,'date'] = expDf.loc[:,'date'].apply(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2][2:])  
        
    #### 1.9 Format activation fields
    try:
        expDf['first activation'] = expDf['first activation'].astype(np.float)
        expDf['activation frequency'] = expDf['activation frequency'].astype(np.float)
    except:
        pass

    #### 2. Save the table, if required
    if save:
        saveName = 'ExperimentalConditions' + suffix + '.csv'
        savePath = os.path.join(DirExp, saveName)
        expDf.to_csv(savePath, sep=';', index = False)
        
        if not cp.CloudSaving == '':
            savePath_cloud = os.path.join(cp.DirCloudExp, saveName)
            expDf.to_csv(savePath_cloud, sep=';', index = False)

    #### 3. Generate additionnal field that won't be saved
    
    #### 3.1 Make 'manipID'
    expDf['manipID'] = expDf['date'] + '_' + expDf['manip']
    
    # def str2int(s):
    #     try:
    #         x = int(s)
    #     except:
    #         x = np.nan
    #     return(x)
    
    # def str2float(s):
    #     try:
    #         x = float(s)
    #     except:
    #         x = np.nan
    #     return(x)
    
    
    # #### 3.2 Format 'bead diameter'
    # diameters = expDf.loc[:,'bead diameter'].apply(lambda x: str(x).split('_'))
    # diameters = diameters.apply(lambda x: [int(xx) for xx in x])
    # expDf.loc[:,'bead diameter'] = diameters
    # # print(ORANGE + 'ramp field : converted to list successfully' + NORMAL)
    
    # #### 3.3 Format 'bead type'
    # bt = expDf.loc[:,'bead type'].apply(lambda x: str(x).split('_'))
    # bt = bt.apply(lambda x: [str(xx) for xx in x])
    # expDf.loc[:,'bead type'] = bt
    
    # #### 3.4 Format 'ramp field'
    # rf = expDf.loc[:,'ramp field'].apply(lambda x: str(x).split('_'))
    # rf = rf.apply(lambda x: [str2float(xx) for xx in x])
    # expDf.loc[:,'ramp field'] = rf
    
    # #### 3.5 Format 'loop structure'
    # ls = expDf.loc[:,'loop structure'].apply(lambda x: str(x).split('_'))
    # ls = ls.apply(lambda x: [str2int(xx) for xx in x])
    # expDf.loc[:,'loop structure'] = ls

    #### 4. END
    return(expDf)



def correctExcelDatesInDf(df, dateColumn, dateExample = ''):
    if dateExample == '':
        dateExample = df.loc[df.index[1], dateColumn]
    if re.match(dateFormatExcel, dateExample):
        print(gs.ORANGE + 'dates : format corrected' + gs.NORMAL)
        df.loc[:,dateColumn] = df.loc[:,dateColumn].apply(lambda x: x.split('/')[0] + '-' + x.split('/')[1] + '-' + x.split('/')[2][2:])        
    elif re.match(dateFormatExcel2, dateExample):
        print(gs.ORANGE + 'dates : format corrected' + gs.NORMAL)
        df.loc[:,dateColumn] = df.loc[:,dateColumn].apply(lambda x: x.split('-')[0] + '-' + x.split('-')[1] + '-' + x.split('-')[2][2:])
    return(df)


def removeColumnsDuplicate(df):
    cols = df.columns.values
    for c in cols:
        if c.endswith('_x'):
            df = df.rename(columns={c: c[:-2]})
        elif c.endswith('_y'):
            df = df.drop(columns=[c])
    return(df)


def findActivation(fieldDf):
    maxZidx = fieldDf['Z'].argmax() #Finding the index of the max Z
    maxZ = fieldDf['Z'][maxZidx] #To check if the value is correct
    return(maxZidx, maxZ)   



def findInfosInFileName(f, infoType):
    """
    Return a given type of info from a file name.
    Inputs : f (str), the file name.
             infoType (str), the type of info wanted.
             infoType can be equal to : 
             * 'M', 'P', 'C' -> will return the number of manip (M), well (P), or cell (C) in a cellID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'C', the function will return 8.
             * 'manipID'     -> will return the full manip ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'manipID', the function will return '21-01-18_M2'.
             * 'cellID'     -> will return the full cell ID.
             ex : if f = '21-01-18_M2_P1_C8.tif' and infoType = 'cellID', the function will return '21-01-18_M2_P1_C8'.
    """
    infoString = ''
    try:
        if infoType in ['M', 'P', 'C']:
            acceptedChar = [str(i) for i in range(10)] + ['-']
            string = '_' + infoType
            iStart = re.search(string, f).end()
            i = iStart
            infoString = '' + f[i]
            while f[i+1] in acceptedChar and i < len(f)-1:
                i += 1
                infoString += f[i]
                
        elif infoType == 'date':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date
        
        elif infoType == 'manipID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            manip = 'M' + findInfosInFileName(f, 'M')
            infoString = date + '_' + manip
            
        elif infoType == 'cellName':
            infoString = 'M' + findInfosInFileName(f, 'M') + \
                         '_' + 'P' + findInfosInFileName(f, 'P') + \
                         '_' + 'C' + findInfosInFileName(f, 'C')
            
        elif infoType == 'cellID':
            datePos = re.search(r"[\d]{1,2}-[\d]{1,2}-[\d]{2}", f)
            date = f[datePos.start():datePos.end()]
            infoString = date + '_' + 'M' + findInfosInFileName(f, 'M') + \
                                '_' + 'P' + findInfosInFileName(f, 'P') + \
                                '_' + 'C' + findInfosInFileName(f, 'C')
                                
        elif infoType == 'substrate':
            try:
                pos = re.search(r"disc[\d]*um", f)
                infoString = f[pos.start():pos.end()]
            except:
                infoString = ''
    except:
        pass
                             
    return(infoString)


# def isFileOfInterest_V0(f, manips, wells, cells):
#     """
#     Determine if a file f correspond to the given criteria.
#     More precisely, return a boolean saying if the manip, well and cell number are in the given range.
#     f is a file name. Each of the fields 'manips', 'wells', 'cells' can be either a number, a list of numbers, or 'all'.
#     Example : if f = '21-01-18_M2_P1_C8.tif'
#     * manips = 'all', wells = 'all', cells = 'all' -> the function return True.
#     * manips = 1, wells = 'all', cells = 'all' -> the function return False.
#     * manips = [1, 2], wells = 'all', cells = 'all' -> the function return True.
#     * manips = [1, 2], wells = 2, cells = 'all' -> the function return False.
#     * manips = [1, 2], wells = 1, cells = [5, 6, 7, 8] -> the function return True.
#     Note : if manips = 'all', the code will consider that wells = 'all', cells = 'all'.
#            if wells = 'all', the code will consider that cells = 'all'.
#            This means you can add filters only in this order : manips > wells > cells.
#     """
#     test = False
#     if f.endswith(".tif"):
#         if manips == 'all':
#             test = True
#         else:
#             try:
#                 manips_str = [str(i) for i in manips]
#             except:
#                 manips_str = [str(manips)]
#             infoM = findInfosInFileName(f, 'M')
#             if infoM in manips_str:
#                 if wells == 'all':
#                     test = True
#                 else:
#                     try:
#                         wells_str = [str(i) for i in wells]
#                     except:
#                         wells_str = [str(wells)]
#                     infoP = findInfosInFileName(f, 'P')
#                     if infoP in wells_str:
#                         if cells == 'all':
#                             test = True
#                         else:
#                             try:
#                                 cells_str = [str(i) for i in cells]
#                             except:
#                                 cells_str = [str(cells)]
#                             infoC = findInfosInFileName(f, 'C')
#                             if infoC in cells_str:
#                                 test = True
#     return(test)



def isFileOfInterest(f, manips, wells, cells):
    """
    Determine if a file f correspond to the given criteria.
    More precisely, return a boolean saying if the manip, well and cell number are in the given range.
    f is a file name. Each of the fields 'manips', 'wells', 'cells' can be either a number, a list of numbers, or 'all'.
    Example : if f = '21-01-18_M2_P1_C8.tif'
    * manips = 'all', wells = 'all', cells = 'all' -> the function return True.
    * manips = 1, wells = 'all', cells = 'all' -> the function return False.
    * manips = [1, 2], wells = 'all', cells = 'all' -> the function return True.
    * manips = [1, 2], wells = 2, cells = 'all' -> the function return False.
    * manips = [1, 2], wells = 1, cells = [5, 6, 7, 8] -> the function return True.
    Note : if manips = 'all', the code will consider that wells = 'all', cells = 'all'.
           if wells = 'all', the code will consider that cells = 'all'.
           This means you can add filters only in this order : manips > wells > cells.
    """
    test = False
    testM, testP, testC = False, False, False
    
    try:
        fM = int(findInfosInFileName(f, 'M'))
        fP = int(findInfosInFileName(f, 'P'))
        fC = int(findInfosInFileName(f, 'C'))
    except:
        return(False)
        
    # print(fM, fP, fC)
    # print(manips, wells, cells)
    # print(toList(manips), toList(wells), toList(cells))
    
    if (manips == 'all') or (fM in toList(manips)):
        testM = True
    if (wells == 'all') or (fP in toList(wells)):
        testP = True
    if (cells == 'all') or (fC in toList(cells)):
        testC = True
        
    if testM and testP and testC:
        test = True
    
    return(test)

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

# def getDictAggMean_V0(df):
#     dictAggMean = {}
#     for c in df.columns:
#     #         t = df[c].dtype
#     #         print(c, t)
#             try :
#                 if np.array_equal(df[c], df[c].astype(bool)):
#                     dictAggMean[c] = 'min'
#                 else:
#                     try:
#                         if not c.isnull().all():
#                             np.mean(df[c])
#                             dictAggMean[c] = 'mean'
#                     except:
#                         dictAggMean[c] = 'first'
#             except:
#                     dictAggMean[c] = 'first'
#     return(dictAggMean)


def archiveData(df, name = '', sep = ';', descText = '',
                saveDir = '', subDir = '', cloudSave = 'none'):
    # Generate unique name if needed
    if name == '':
        dt = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        name = 'df_' + dt
        
    saveDesc = (descText != '')
        
    # Generate default save path if needed
    if saveDir == '':
        saveDir = cp.DirDataFigToday
        saveCloudDir = cp.DirCloudFigToday
    else:
        saveDir = os.path.join(cp.DirDataFig, saveDir)
        saveCloudDir = os.path.join(cp.DirCloudFig, saveDir)
        
    # Normal save
    savePath = os.path.join(saveDir, subDir, name+'.csv')
    df.to_csv(savePath, sep = sep, index = False)
    if saveDesc:
        descPath = os.path.join(saveDir, subDir, name+'_infos.txt')
        f = open(descPath, 'w')
        f.write(descText)
        f.close()
    
    # Cloud save if specified
    doCloudSave = (not (cloudSave == 'none')) \
                  or (cloudSave == 'strict') \
                  or (cloudSave == 'flexible' and cp.CloudSaving != '')
                  
    if doCloudSave:
        cloudSavePath = os.path.join(saveCloudDir, subDir, name+'.csv')
        df.to_csv(cloudSavePath, sep = sep, index = False)
        if saveDesc:
            descPathCloud = os.path.join(saveCloudDir, subDir, name+'_infos.txt')
            f = open(descPathCloud, 'w')
            f.write(descText)
            f.close()
        
    



# %%% File manipulation

def copyFile(DirSrc, DirDst, filename):
    """
    Copy the file 'filename' from 'DirSrc' to 'DirDst'
    """
    PathSrc = os.path.join(DirSrc, filename)
    PathDst = os.path.join(DirDst, filename)
    shutil.copyfile(PathSrc, PathDst)
    
def copyFolder(DirSrc, DirDst, folderName):
    """
    Copy the folder 'folderName' from 'DirSrc' to 'DirDst' with all its contents
    """
    PathSrc = os.path.join(DirSrc, folderName)
    PathDst = os.path.join(DirDst, folderName)
    shutil.copytree(PathSrc, PathDst)
    
def copyFilesWithString(DirSrc, DirDst, stringInName):
    """
    Copy all the files from 'DirSrc' which names contains 'stringInName' to 'DirDst'
    """
    SrcFilesList = os.listdir(DirSrc)
    for SrcFile in SrcFilesList:
        if stringInName in SrcFile:
            copyFile(DirSrc, DirDst, SrcFile)
            
def containsFilesWithExt(Dir, ext):
    answer = False
    FilesList = os.listdir(Dir)
    for File in FilesList:
        if File.endswith(ext):
            answer = True
    return(answer)

def softMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %%% Stats

def get_R2(Y1, Y2):
    meanY = np.mean(Y1)
    meanYarray = meanY*np.ones(len(Y1))
    SST = np.sum((Y1-meanYarray)**2)
    SSE = np.sum((Y2-meanYarray)**2)
    R2 = SSE/SST
    return(R2)

def get_Chi2(Ymeas, Ymodel, dof, S):
    residuals = Ymeas-Ymodel
    Chi2 = np.sum((residuals/S)**2)
    Chi2_dof = Chi2/dof
    return(Chi2_dof)

# %%% Image processing

def getDepthoCleanSize(D, scale):
    """
    Function that looks stupid but is quite important ! It allows to standardise 
    across all other functions the way the depthograph width is computed.
    D here is the approximative size of the bead in microns, 4.5 for M450, 2.7 for M270.
    Scale is the pixel to microns ration of the objective.
    """
    cleanSize = int(np.floor(1*D*scale))
    cleanSize += 1 + cleanSize%2
    return(cleanSize)

def compute_cost_matrix(XY1,XY2):
    """
    Compute a custom cost matrix between two arrays of XY positions.
    Here the costs are simply the squared distance between each XY positions.
    Example : M[2,1] is the sqaured distance between XY1[2] and XY2[1], 
    which is ((XY2[1,1]-XY1[2,1])**2 + (XY2[1,0]-XY1[2,0])**2)
    """
    N1, N2 = XY1.shape[0],XY2.shape[0]
    M = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            M[i,j] = (np.sum((XY2[j,:] - XY1[i,:]) ** 2))
    return(M)

def ui2array(uixy):
    """
    Translate the output of the function plt.ginput() 
    (which are lists of tuples), in an XY array with this shape:
    XY = [[x0, y0], [x1, y1], [x2, y2], ...]
    So if you need the [x, y] of 1 specific point, call XY[i]
    If you need the list of all x coordinates, call XY[:, 0]
    """
    n = len(uixy)
    XY = np.zeros((n, 2))
    for i in range(n):
        XY[i,0], XY[i,1] = uixy[i][0], uixy[i][1]
    return(XY)

def getROI(roiSize, x0, y0, nx, ny):
    """
    Return coordinates of top left (x1, y1) and bottom right (x2, y2) corner of a ROI, 
    and a boolean validROI that says if the ROI exceed the limit of the image.
    Inputs : 
    - roiSize, the width of the (square) ROI.
    - x0, y0, the position of the central pixel.
    - nx, ny, the size of the image.
    Note : the ROI is done so that the final width (= height) 
    of the ROI will always be an odd number.
    """
    roiSize += roiSize%2
    x1 = int(np.floor(x0) - roiSize*0.5) - 1
    x2 = int(np.floor(x0) + roiSize*0.5)
    y1 = int(np.floor(y0) - roiSize*0.5) - 1
    y2 = int(np.floor(y0) + roiSize*0.5)
    if min([x1,nx-x2,y1,ny-y2]) < 0:
        validROI = False
    else:
        validROI = True
    return(x1, y1, x2, y2, validROI)



def squareDistance(M, V, normalize = False): # MUCH FASTER ! **Michael Scott Voice** VERRRRY GOODE
    """
    Compute a distance between two arrays of the same size, defined as such:
    D = integral of the squared difference between the two arrays.
    It is used to compute the best fit of a slice of a bead profile on the depthograph.
    This function speed is critical for the Z computation process because it is called so many times !
    What made that function faster is the absence of 'for' loops and the use of np.repeat().
    """
    #     top = time.time()
    n, m = M.shape[0], M.shape[1]
    # len(V) should be m
    if normalize:
        V = V/np.mean(V)
    V = np.array([V])
    MV = np.repeat(V, n, axis = 0) # Key trick for speed !
    if normalize:
        M = (M.T/np.mean(M, axis = 1).T).T
    R = np.sum((M-MV)**2, axis = 1)
#     print('DistanceCompTime')
#     print(time.time()-top)
    return(R)

def matchDists(listD, listStatus, Nup, NVox, direction):
    """
    This function transform the different distances curves computed for 
    a Nuplet of images to match their minima. By definition it is not used for singlets of images.
    In practice, it's a tedious and boring function.
    For a triplet of image, it will move the distance curve by NVox voxels to the left 
    for the first curve of a triplet, not move the second one, and move the third by NVox voxels to the right.
    The goal : align the 3 matching minima so that the sum of the three will have a clear global minimum.
    direction = 'upward' or 'downward' depending on how your triplet images are taken 
    (i.e. upward = consecutively towards the bright spot and downwards otherwise)
    """
    N = len(listStatus)
    offsets = np.array(listStatus) - np.ones(N) * (Nup//2 + 1)
    offsets = offsets.astype(int)
    listD2 = []
    if direction == 'upward':
        for i in range(N):
            if offsets[i] < 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((D[shift:],fillVal*np.ones(shift))).astype(np.float64)
                listD2.append(D2)
            if offsets[i] == 0:
                D = listD[i].astype(np.float64)
                listD2.append(D)
            if offsets[i] > 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((fillVal*np.ones(shift),D[:-shift])).astype(np.float64)
                listD2.append(D2)
    elif direction == 'downward':
        for i in range(N):
            if offsets[i] > 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((D[shift:],fillVal*np.ones(shift))).astype(np.float64)
                listD2.append(D2)
            if offsets[i] == 0:
                D = listD[i].astype(np.float64)
                listD2.append(D)
            if offsets[i] < 0:
                shift = abs(offsets[i])*NVox
                D = listD[i]
                fillVal = max(D)
                D2 = np.concatenate((fillVal*np.ones(shift),D[:-shift])).astype(np.float64)
                listD2.append(D2)
    return(np.array(listD2))


def uiThresholding(I, method = 'otsu', factorT = 0.8):
    """
    Interactive thresholding function to replace IJ.
    Compute an auto thresholding on a global 3D image with a method from this list:
    > 'otsu', 'max_entropy', (add the method you want ! here are the options : https://scikit-image.org/docs/stable/api/skimage.filters.html )
    Then display a figure for the user to assess the threshold fitness, and according to the user choice,
    confirm the threshold or recompute it with new parameters in a recursive way.
    """
    # 1. Compute the threshold
    nz = I.shape[0]
    if method == 'otsu':
        threshold = factorT*filters.threshold_otsu(I)
    elif method == 'max_entropy':
        bitDepth = util.dtype_limits(I)[1]+1
        I8 = util.img_as_ubyte(I)
        threshold = factorT*max_entropy_threshold(I8)*(bitDepth/2**8)
        
    # 2. Display images for the user to assess the fitness
    # New version of the plot
        nS = I.shape[0]
        loopSize = nS//4
        N = min(4, nS//loopSize)
        L_I_plot = [I[loopSize*2*k + 2] for k in range(N)]
        L_I_thresh = [I_plot > threshold for I_plot in L_I_plot]
        for i in range(N):
            I_plot = L_I_plot[i]
            I_thresh = L_I_thresh[i]
            I_plot = util.img_as_ubyte(I_plot)
            I_plot = color.gray2rgb(I_plot)
            pStart, pStop = np.percentile(I_plot, (1, 99))
            I_plot = exposure.rescale_intensity(I_plot, in_range=(pStart, pStop))
            red_multiplier = [255, 0, 0]
            I_plot[I_thresh] = red_multiplier
            L_I_plot[i] = I_plot
            
        I_thresh_all = I > threshold
        I_thresh_max = np.max(I_thresh_all, axis = 0)
        
        fig = plt.figure(tight_layout=True)
        gs = GridSpec(2, 4, figure=fig)
        ax = []
        for i in range(N):
            ax.append(fig.add_subplot(gs[i//2, i%2]))
            ax[-1].imshow(L_I_plot[i])
            ax[-1].set_title('Frame ' + str(loopSize*2*i + 2) + '/' + str(nS), fontsize = 8)
            ax[-1].axes.xaxis.set_ticks([])
            ax[-1].axes.yaxis.set_ticks([])
        ax.append(fig.add_subplot(gs[:, 2:]))
        ax[-1].imshow(I_thresh_max, cmap = 'gray')
        ax[-1].set_title('Max projection', fontsize = 10)
        ax[-1].axes.xaxis.set_ticks([])
        ax[-1].axes.yaxis.set_ticks([])
        fig.suptitle(str(threshold), fontsize = 12)
        fig.show()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(50, 380, 1800, 650)
    
    # 3. Ask the question to the user
    QA = pyautogui.confirm(
                text='Is the threshold satisfying?',
                title='Confirm threshold', 
                buttons=['Yes', '10% Lower', '5% Lower', '1% Lower', '1% Higher', '5% Higher', '10% Higher'])
    plt.close(fig)
    
    # 4. Recall the same function with new parameters, or validate the threshold 
    # according to the user answer.
    increment = 0.1 * ('10%' in QA) + 0.05 * ('5%' in QA) + 0.01 * ('1%' in QA)
    if 'Lower' in QA:
        uiThresholding(method = method, factorT = factorT - increment)
    elif 'Higher' in QA:
        uiThresholding(method = method, factorT = factorT + increment)
    elif QA == 'Yes':
        threshold = threshold
    return(threshold)

def max_entropy(data):
    """
    Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method
    Kapur J.N., Sahoo P.K., and Wong A.K.C. (1985) "A New Method for Gray-Level Picture Thresholding Using the Entropy
    of the Histogram", Graphical Models and Image Processing, 29(3): 273-285
    M. Emre Celebi
    06.15.2007
    Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
    2016-04-28: Adapted for Python 2.7 by Robert Metchev from Java source of MaxEntropy() in the Autothresholder plugin
    http://rsb.info.nih.gov/ij/plugins/download/AutoThresholder.java
    :param data: Sequence representing the histogram of the image
    :return threshold: Resulting maximum entropy threshold
    """

    # calculate CDF (cumulative density function)
    cdf = data.astype(np.float).cumsum()

    # find histogram's nonzero area
    valid_idx = np.nonzero(data)[0]
    first_bin = valid_idx[0]
    last_bin = valid_idx[-1]

    # initialize search for maximum
    max_ent, threshold = 0, 0

    for it in range(first_bin, last_bin + 1):
        # Background (dark)
        hist_range = data[:it + 1]
        hist_range = hist_range[hist_range != 0] / cdf[it]  # normalize within selected range & remove all 0 elements
        tot_ent = -np.sum(hist_range * np.log(hist_range))  # background entropy

        # Foreground/Object (bright)
        hist_range = data[it + 1:]
        # normalize within selected range & remove all 0 elements
        hist_range = hist_range[hist_range != 0] / (cdf[last_bin] - cdf[it])
        tot_ent -= np.sum(hist_range * np.log(hist_range))  # accumulate object entropy

        # find max
        if tot_ent > max_ent:
            max_ent, threshold = tot_ent, it

    return(threshold)

def max_entropy_threshold(I):
    """
    Function based on the previous one that directly takes an image for argument.
    """
    H, bins = exposure.histogram(I, nbins=256, source_range='image', normalize=False)
    T = max_entropy(H)
    return(T)

# %%% Physics

def computeMag_M270(B):
    M = 0.74257*1.05*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
    return(M)

def computeMag_M450(B):
    M = 1.05*1600 * (0.001991*B**3 + 17.54*B**2 + 153.4*B) / (B**2 + 35.53*B + 158.1)
    return(M)

def chadwickModel(h, E, H0, DIAMETER):
    R = DIAMETER/2
    f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    return(f)

def inversedChadwickModel(f, E, H0, DIAMETER):
    R = DIAMETER/2
    h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    return(h)

def getDimitriadisCoefs(v, order):
    a0 = -0.347*(3 - 2*v)/(1 - v)
    b0 = 0.056*(5 - 2*v)/(1 - v)
    
    ks = []
    if order >= 0:
        k0 = 1
        ks.append(k0)
        
        if order >= 1:
            k1 = - 2*a0/np.pi
            ks.append(k1)
        
            if order >= 2:
                k2 = + 4*a0**2/np.pi
                ks.append(k2)
            
                if order >= 3:
                    k3 = - (8/np.pi**3) * (a0**3 + (4*np.pi**2/15)*b0)
                    ks.append(k3)
            
                    if order >= 4:
                        k4 = + (16*a0/np.pi**4) * (a0**3 + (3*np.pi**2/5)*b0)
                        ks.append(k4)    
    return(ks)


def dimitriadisModel(h, E, H0):
    DIAMETER = 4477
    R = DIAMETER/2
    
    v = 0
    order = 2
    ks = getDimitriadisCoefs(v, order)
    
    delta = H0-h
    X = np.sqrt(R*delta)/h
    
    poly = np.zeros_like(X)
    
    for i in range(order+1):
        poly = poly + ks[i] * X**i
        
    F = ((4 * E * R**0.5 * delta**1.5)/(3 * (1 - v**2))) * poly
    
    return(F)
 
                    
# %%% Very general functions

def findFirst(x, A):
    """
    Find first occurence of x in array A, in a VERY FAST way.
    If you like weird one liners, you will like this function.
    """
    idx = (A==x).view(bool).argmax()
    return(idx)

def fitLine(X, Y):
    """
    returns: results.params, results \n
    Y=a*X+b ; params[0] = b,  params[1] = a
    
    NB:
        R2 = results.rsquared \n
        ci = results.conf_int(alpha=0.05) \n
        CovM = results.cov_params() \n
        p = results.pvalues \n
    
    This is how one should compute conf_int:
        bse = results.bse \n
        dist = stats.t \n
        alpha = 0.05 \n
        q = dist.ppf(1 - alpha / 2, results.df_resid) \n
        params = results.params \n
        lower = params - q * bse \n
        upper = params + q * bse \n
    """
    
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    params = results.params 
#     print(dir(results))
    return(results.params, results)

def toList(x):
    """
    if x is a list, return x
    if x is a number, return [x]
    """
    try:
        x = list(x)
        return(x)
    except:
        return([x])
    
def drop_duplicates_in_array(A):
    val, idx = np.unique(A, return_index = True)
    idx.sort()
    A_filtered = np.array([A[idx[j]] for j in range(len(idx))])
    return(A_filtered)

# %%% Figure & graphic operations

def simpleSaveFig(fig, name, savePath, ext, dpi):
    figPath = os.path.join(savePath, name + ext)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    fig.savefig(figPath, dpi=dpi)
    

def archiveFig(fig, name = '', ext = '.png', dpi = 100,
               figDir = '', figSubDir = '', cloudSave = 'flexible'):
    """
    This is supposed to be a "smart" figure saver. \n
    (1) \n
    - It saves the fig with resolution 'dpi' and extension 'ext' (default ext = '.png' and dpi = 100). \n
    - If you give a name, it will be used to save your file; if not, a name will be generated based on the date. \n
    - If you give a value for figDir, your file will be saved in cp.DirDataFig//figDir. Else, it will be in cp.DirDataFigToday. \n
    - You can also give a value for figSubDir to save your fig in a subfolder of the chosen figDir. \n
    (2) \n
    cloudSave can have 3 values : 'strict', 'flexible', or 'none'. \n
    - If 'strict', this function will attempt to do a cloud save not matter what. \n
    - If 'check', this function will check that you enable properly the cloud save in CortexPath before attempting to do a cloud save. \n
    - If 'none', this function will not do a cloud save. \n
    """
    # Generate unique name if needed
    if name == '':
        dt = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        name = 'fig_' + dt
    # Generate default save path if needed
    if figDir == '':
        figDir = cp.DirDataFigToday
        figCloudDir = cp.DirCloudFigToday
    else:
        figDir = os.path.join(cp.DirDataFig, figDir)
        figCloudDir = os.path.join(cp.DirCloudFig, figDir)
    # Normal save
    savePath = os.path.join(figDir, figSubDir)
    simpleSaveFig(fig, name, savePath, ext, dpi)
    # Cloud save if specified
    doCloudSave = ((cloudSave == 'strict') or (cloudSave == 'flexible' and cp.CloudSaving != ''))
    if doCloudSave:
        cloudSavePath = os.path.join(figCloudDir, figSubDir)
        simpleSaveFig(fig, name, cloudSavePath, ext, dpi)
        
   
    

# def archiveFig_V0(fig, ax, figDir, name='auto', dpi = 100):
    
#     if not os.path.exists(figDir):
#         os.makedirs(figDir)
    
#     # saveDir = os.path.join(figDir, str(date.today()))
#     # if not os.path.exists(saveDir):
#     #     os.makedirs(saveDir)
#     saveDir = figDir
    
#     if name != 'auto':
#         fig.savefig(os.path.join(saveDir, name + '.png'), dpi=dpi)
    
#     else:
#         suptitle = fig._suptitle.get_text()
#         if len(suptitle) > 0:
#             name = suptitle
#             fig.savefig(os.path.join(saveDir, name + '.png'), dpi=dpi)
        
#         else:
#             try:
#                 N = len(ax)
#                 ax = ax[0]
#             except:
#                 N = 1
#                 ax = ax
                
#             xlabel = ax.get_xlabel()
#             ylabel = ax.get_ylabel()
#             if len(xlabel) > 0 and len(ylabel) > 0:
#                 name = ylabel + ' Vs ' + xlabel
#                 if N > 1:
#                     name = name + '___etc'
#                 fig.savefig(os.path.join(saveDir, name + '.png'), dpi=dpi)
            
#             else:
#                 title = ax.get_title()
#                 if len(title) > 0:
#                     if N > 1:
#                         name = name + '___etc'
#                     fig.savefig(os.path.join(saveDir, name + '.png'), dpi=dpi)
                
#                 else:
#                     figNum = plt.gcf().number
#                     name = 'figure ' + str(figNum) 
#                     fig.savefig(os.path.join(saveDir, name + '.png'), dpi=dpi)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]))