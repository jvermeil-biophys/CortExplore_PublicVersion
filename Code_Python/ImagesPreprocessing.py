# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:21:10 2022
@author: Anumita Jawahar & Joseph Vermeil

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

source : https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
"""

#### Imports

import os
import cv2
import shutil
import traceback

import numpy as np
import pyjokes as pj

from skimage import io

#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import GlobalConstants as gc
import UtilityFunctions as ufun

#%% Define parameters # Numi

# date = '22.08.26'
# DirSave = os.path.join(cp.DirDataRaw, date)
# DirExt = 'G:/20220826_100xoil_3t3optorhoa_4.5beadsStrept_Mechanics/22.08.26'
# # prefix = 'cell'
# # channel = 'w1TIRF DIC'
# microscope = 'labview'

#%% Define parameters # Jojo

date = '22.03.28'
DirExt = 'E:/22.03.28_Patterns3T3_drugs' #'/M4_patterns_ctrl'
DirSave = os.path.join(cp.DirDataRaw, date)

prefix = ''
channel = ''
microscope = 'labview'


# %% Functions

def getListOfSourceFolders(Dir, forbiddenWords = ['deptho', 'error', 'excluded', 'out', 'bad']):
    """
    Given a root folder Dir, search recursively inside for all folders containing .tif images 
    and whose name do not contains any of the forbiddenWords.
    """
    
    res = []
    exclude = False
    for w in forbiddenWords:
        if w.lower() in Dir.lower(): # compare the lower case strings
            exclude = True # If a forbidden word is in the dir name, don't consider it
            
            
    if exclude or not os.path.isdir(Dir):
        return(res) # Empty list
    
    elif ufun.containsFilesWithExt(Dir, '.tif'):
        res = [Dir] # List with 1 element - the name of this dir
        
    else:
        listDirs = os.listdir(Dir)
        for D in listDirs:
            path = os.path.join(Dir, D)
            res += getListOfSourceFolders(path) # Recursive call to the function !
    # In the end this function will have explored all the sub directories of Dir,
    # searching for folders containing tif files, without forbidden words in their names.        
    return(res)

# test

# testDir = 'G:/22.07.15_longLinker'
# A = getListOfSourceFolders(testDir)

#

def copyFieldFiles(ListDirSrc, DirDst, suffix = '_Field.txt'):
    """
    Import the Field.txt files from the relevant folders.
    Calls the copyFilesWithString from ufun with suffix = '_Field.txt'
    
    """
    
    for DirSrc in ListDirSrc:
        ufun.copyFilesWithString(DirSrc, DirDst, suffix)


def removeFrames(DirSave):
    """
    Remove some extra frames from all files that can sometimes (quite rarely) arise from Labview 
    bugging out and interfere with tracking in MainTracker.
    Can even be used for metamorph files.
    Example: an incomplete triplet
    
    """
    
    allFiles = os.listdir(DirSave)
    for f in allFiles:
        if f.endswith('.tif'):
            print('Loading..'+f)
            ic = io.ImageCollection(DirSave+'/'+f, conserve_memory = True)
            stack = io.concatenate_images(ic[:4500])
            print('Saving...'+f)
            io.imsave(DirSave+'/'+f, stack)


def AllMMTriplets2Stack(DirExt, DirSave, prefix, channel):
    """
    Used for metamorph created files.
    Metamoprh does not save images in stacks but individual triplets. These individual triplets take time
    to open in FIJI.
    This function takes images of a sepcific channel and creates .tif stacks from them.
       
    """
    
    allCells = os.listdir(DirExt)
    for currentCell in allCells:
        path = os.path.join(DirExt, currentCell)
        allFiles = os.listdir(path)
        date = ufun.findInfosInFileName(currentCell, 'date')
        allFiles = [path+'/'+string for string in allFiles if channel in string]
        #+4 at the end corrosponds to the '_t' part to sort the array well
        limiter = len(path)+len(prefix)+len(channel)+4 
        allFiles.sort(key=lambda x: int(x[limiter:-4]))
        ic = io.ImageCollection(allFiles, conserve_memory = True)
        stack = io.concatenate_images(ic)
        if '561' in channel:
            date = date.replace('-', '.')
            folderName = date+'_561'
            fluoPath = os.path.join(DirSave, folderName)
            try:
                os.mkdir(fluoPath)
            except:
                pass
            
        io.imsave(fluoPath+'/'+currentCell+'.tif', stack)
        
        
def renamePrefix(DirExt, currentCell, newPrefix):
    """
    Used for metamorph created files.
    Metamorph creates a new folder for each timelapse, within which all images contain a predefined 
    'prefix' and 'channel' name which can differ between microscopes. Eg.: 'w1TIRF_DIC' or 'w2TIRF_561'
    
    If you forget to create a new folder for a new timelapse, Metamorph automatically changes the prefix
    to distinguish between the old and new timelapse triplets. This can get annoying when it comes to processing 
    many cells.
    
    This function allows you to rename the prefix of all individual triplets in a specific folder. 
    
    """
    
    path = os.path.join(DirExt, currentCell)
    allImages = os.listdir(path)
    for i in allImages:
        if i.endswith('.TIF'):
            split = i.split('_')
            split[0] = newPrefix
            newName = '_'.join()
            os.rename(os.path.join(path,i), os.path.join(path, newName))
            

def renameCells(DirSave):
    """
    Used for metamorph created files.
    Metamorph creates a new folder for each timelapse, within which all images contain a predefined 
    'prefix' and 'channel' name which can differ between microscopes. Eg.: 'w1TIRF_DIC' or 'w2TIRF_561'
    
    If you forget to create a new folder for a new timelapse, Metamorph automatically changes the prefix
    to distinguish between the old and new timelapse triplets. This can get annoying when it comes to processing 
    many cells.
    
    This function allows you to rename the prefix of all individual triplets in a specific folder. 
    
    """
    
    allImages = os.listdir(DirSave)
    for i in allImages:
        if i.endswith('_active_Field.txt'):
            split = i.split('_')
            if split[1] == 'M1':
                print(i)
                split[1] = 'M5'
                newName = '_'.join(split[:5])+'_L40_Field.txt'
            # elif split[1] == 'M3':
            #     print(i)
            #     split[1] = 'M7'
            #     newName = '_'.join(split[:5])+'_L40_Field.txt'
            # elif split[1] == 'M4':
            #     print(i)
            #     split[1] = 'M8'
            #     newName = '_'.join(split[:5])+'_L40_Field.txt'
                os.rename(os.path.join(DirSave,i), os.path.join(DirSave, newName))
            

def Zprojection(currentCell, microscope, kind = 'min', channel = 'nan', prefix = 'nan'):
    """
    From an image stack contained in 'currentCell' folder,
    does a scaled-down (by 'scaleFactor') Z-projection (minimum by default)
    to display the best image for cropping boundary selection.
    
    If you are using Metamorph for imaging, you will have to update the 'prefix' and 'channel'
    variables depending on the names of your images. 
    It follows the usual, default metamorph naming system: 'Prefix_Channel_Timepoint0.tif'
    If you are using labview, the default will be 'nan' for both variables.
    """
    
    scaleFactor = 4
    path = os.path.join(DirExt, currentCell)
    allFiles = os.listdir(path)
    
    if microscope == 'metamorph':
        allFiles = [path+'/'+string for string in allFiles if channel in string]
        #+4 at the end corrosponds to the '_t' part to sort the array well
        limiter = len(path)+len(prefix)+len(channel)+4 
        allFiles.sort(key=lambda x: int(x[limiter:-4]))
    
    elif microscope == 'labview':
        allFiles = [path+'/'+string for string in allFiles if 'im' in string]
        
    idx = slice(0, len(allFiles), 100)
    
    allFiles = allFiles[idx]
    frame = cv2.imread(allFiles[0])
    imgWidth, imgHeight = frame.shape[1], frame.shape[0]
    ic = io.ImageCollection(allFiles, conserve_memory=True)
    stack = io.concatenate_images(ic)
        
    if kind == 'min':
        Zimg = np.min(stack, axis = 0)
    elif kind == 'max':
        Zimg = np.max(ic, axis = 0)
        
    Zimg = cv2.resize(Zimg, (int(imgWidth/scaleFactor), int(imgHeight/scaleFactor)))
    Zimg = cv2.normalize(Zimg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Zimg

def shape_selection(event, x, y, flags, param):
    """
    Non-interactive rectangular selection.
    Has to be called in cv2.setMouseCallback(currentCell, shape_selection)
    """
    
    # grab references to the global variables
    global ref_point, crop, allZimg #, iZ

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [[x, y]]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append([x, y])

        # draw a rectangle around the region of interest
        cv2.rectangle(allZimg[i], ref_point[0], ref_point[1], (0, 255, 0), 1)

  
def shape_selection_V2(event, x, y, flags, param):
    """
    Interactive rectangular selection.
    Has to be called in cv2.setMouseCallback(currentCell, shape_selection_V2)
    """
    
    # Grab references to the global variables
    global ix, iy, drawing, ref_point, crop, img, img_copy
    
    # If the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        img_copy = np.copy(img)
        ref_point = [[x, y]]
    
    # If the mouse moves, reinitialize the image and the rectangle to match 
    # the current position
    elif event == cv2.EVENT_MOUSEMOVE: 
        
        if drawing == True:
            img = np.copy(img_copy)
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append([x, y])
        # Final rectangle around the region of interest
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        

def cropAndCopy(DirSrc, DirDst, allRefPoints, allCellPaths, microscope, channel = 'nan', prefix = 'nan'):
    """
    Using user specified rectangular coordinates from the previous functions,
    Crop cells stack and copy them onto the destination file.
    
    If you are using Metamorph for imaging, you will have to update the 'prefix' and 'channel'
    variables depending on the names of your images. 
    It follows the usual, default metamorph naming system: 'Prefix_Channel_Timepoint0.tif'
    If you are using labview, the default will be 'nan' for both variables.
    
    """
    
    count = 0
    for i in range(len(allCellPaths)):
    # for refPts, cellPath in zip(allRefPoints, allCellPaths):
        
        refPts = np.array(allRefPoints[i])
        
        cellPath = allCellPaths[i]
        cellName = cellPath.split('\\')[-1]
        allFiles = os.listdir(cellPath)
        
        # to detect second selections
        suffix = ''
        try:
            if (allCellPaths[i-1]==allCellPaths[i]):
                suffix = '-1'
        except:
            pass
        
        if microscope == 'metamorph':
            allFiles = [cellPath+'/'+string for string in allFiles if channel in string]
            #+4 at the end corrosponds to the '_t' part to sort the array well
            limiter = len(cellPath)+len(prefix)+len(channel)+4 
            allFiles.sort(key=lambda x: int(x[limiter:-4]))
    
        elif microscope == 'labview':
            allFiles = [cellPath+'/'+string for string in allFiles if 'im' in string]
        
        print(gs.BLUE + 'Loading '+ cellPath +'...' + gs.NORMAL)
        
        ic = io.ImageCollection(allFiles, conserve_memory = True)
        stack = io.concatenate_images(ic)
        
        x1, x2 = int(min(refPts[:,0])), int(max(refPts[:,0]))
        y1, y2 = int(min(refPts[:,1])), int(max(refPts[:,1]))
        
        # To avoid that the cropped region gets bigger than the image itself
        ny, nx = stack.shape[1], stack.shape[2]
        x1, x2, y1, y2 = max(0, x1), min(nx, x2), max(0, y1), min(ny, y2)
        
        
        try:
            cropped = stack[:, y1:y2, x1:x2]
            io.imsave(DirDst + '/' + cellName + suffix + '.tif', cropped)
            print(gs.GREEN + DirDst + '/' + cellName + '.tif' + '\nSaved sucessfully' + gs.NORMAL)
        
        except Exception:
            traceback.print_exc()
            print(gs.RED + DirDst + '/' + cellName + '.tif' + '\nError when saving' + gs.NORMAL)
            continue
        
        if count%5 == 0:
            joke = pj.get_joke(language='en', category= 'all')
            print(joke)
            count = count + 1




# preprocess(DirExt, DirSave, microscope, reset = 0)    

#%% Main function 1/2

# def preprocess(DirExt, DirSave, microscope, reset = 0):
    
allCellsRaw = getListOfSourceFolders(DirExt)[:]
allCells = []
allCellsToCrop = []
ref_point = []
allRefPoints = []
allZimg = []
allZimg_og = []


# reset = 0
checkIfAlreadyExist = True

if not os.path.exists(DirSave):
    os.mkdir(DirSave)

print(gs.BLUE + 'Constructing all Z-Projections...' + gs.NORMAL)

scaleFactor = 4

for i in range(len(allCellsRaw)):
    print(i)
    currentCell = allCellsRaw[i]
    currentCellName = currentCell.split('\\')[-1]
    validCell = True
    print(currentCellName)
        
    if not ufun.containsFilesWithExt(currentCell, '.tif'):
        validCell = False
        print(gs.BRIGHTRED + '/!\ Is not a valid cell' + gs.NORMAL)
        
    elif checkIfAlreadyExist and os.path.isfile(os.path.join(DirSave, currentCellName + '.tif')):
        validCell = False
        print(gs.GREEN + ':-) Has already been copied' + gs.NORMAL)
        
    if validCell:
        try:
            Zimg = Zprojection(currentCell, microscope)
            allCells.append(currentCell)
            allZimg.append(Zimg)
            print(gs.CYAN + '--> Will be copied' + gs.NORMAL)
        except:
            print(gs.BRIGHTRED + '/!\ Unexpected error during file handling' + gs.NORMAL)
        
copyFieldFiles(allCells, DirSave)

# allZimg_og = np.copy(np.asarray(allZimg)) # TBC

#%% Main function 2/2

instructionText = "Draw the ROIs to crop !\n\n(1) Click on the image to define a rectangular selection\n"
instructionText += "(2) Press 'a' to accept your selection, 'r' to redraw it, "
instructionText += "or 's' if you have a second selection to make (don't use 'm' more than once per stack)\n"
instructionText += "(3) Make sure to choose the number of files you want to crop at once\nin the variable 'limiter'"
instructionText += "\n\nC'est parti !\n"


#Change below the number of stacks you want to crop at once. Run the code again to crop the remaining files. 
# !!!!!! WARNING: Sometimes choosing too many can make your computer bug!!!!!
limiter = 24

print(gs.YELLOW + instructionText + gs.NORMAL)

# if reset == 1:
#     allZimg = np.copy(allZimg_og)
#     ref_point = []
#     allRefPoints = []

count = 0
# for i in range(len(allZimg)):
for i in range(min(len(allZimg), limiter)):
    if count%24 == 0:
        count = 0
        
    currentCell = allCells[i]
    currentCellName = currentCell.split('\\')[-1]
    
    Nimg = len(allZimg)
    ncols = 6
    nrows = ((Nimg-1) // ncols) + 1
    
    # test
    ix,iy = 0, 0
    drawing = False
    img = allZimg[i]
    img_backup = np.copy(img)
    img_copy = np.copy(img)
    
    cv2.namedWindow(currentCellName)
    cv2.moveWindow(currentCellName, (count//ncols)*400, count%ncols*175)
    
    # cv2.setMouseCallback(currentCell, shape_selection)
    cv2.setMouseCallback(currentCellName, shape_selection_V2)
    
    while True:
    # display the image and wait for a keypress
        currentCell = allCells[i]
        currentCellName = currentCell.split('\\')[-1]
        cv2.imshow(currentCellName, img)
        key = cv2.waitKey(20) & 0xFF
        
    # press 'r' to reset the crop
        if key == ord("r"):
            img = np.copy(img_backup)  
             
    # if the 'a' key is pressed, break from the loop and move on to the next file
        elif key == ord("a"):
            allRefPoints.append(np.asarray(ref_point)*scaleFactor)
            allCellsToCrop.append(currentCell)
            break
        
    # if the 's' key is pressed, save the coordinates and rest the crop, ready to save once more
    # /!\ The code isn't designed to accept more than 2 selections per stack
        elif key == ord("s"):
            allRefPoints.append(np.asarray(ref_point)*scaleFactor)
            allCellsToCrop.append(currentCell)
            img = np.copy(img_backup)
            
        
    count = count + 1
    
cv2.destroyAllWindows()

print(gs.BLUE + 'Saving all tiff stacks...' + gs.NORMAL)

cropAndCopy(DirExt, DirSave, allRefPoints, allCellsToCrop, microscope)


#%% Creating .tif stacks of 561n recruitment images

# DirSave = 'D:/Anumita/MagneticPincherData/Raw/'
# DirExt = 'F:/Cortex Experiments/OptoPincher Experiments/20220322_100xoil_3t3optorhoa_4.5beads_15mT/22.03.22'
# prefix = 'cell'
# channel = 'w3TIRF 561'


# AllMMTriplets2Stack(DirExt, DirSave, prefix, channel)

