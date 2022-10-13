
import numpy as np
cimport numpy as np
import cython
cimport cython
import time, copy
from libc.stdlib cimport srand, rand, RAND_MAX
from libc.math cimport sqrt, exp, log, fmin, M_PI

# from scipy import ndimage
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

DTYPE = np.int64

ctypedef np.int64_t DTYPE_t


#####################################################
###===== For Printing Images =====###
#####################################################
def printImages(int k):    
    path = os.getcwd()
    dirName = 'Results'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    imagefile = 'mcs_'
    joined_image = os.path.join(dirName, imagefile)
    plt.savefig(joined_image + str(k) + '.png', format="PNG", dpi=200, bbox_inches='tight')
##################################################


###########################################
###===== Initial Cell Distribution =====###
###########################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] initializePopulation(list pottsElements, list cellArr):

    cdef:
        int columnlength, rowlength
        int totCells
        int xC, yC
        int i, j

    cdef: 
        int xMax = pottsElements[0]["xDim"]
        int yMax = pottsElements[0]["yDim"]
        int cellsXLattice = cellArr[0]['NumberCellsX']
        int cellsYLattice = cellArr[0]['NumberCellsY']
        int lengthCellX = cellArr[1]["LengthCellX"]
        int lengthCellY = cellArr[1]["LengthCellY"]
        int gap = cellArr[2]["Gap"]
        int xCent = cellArr[3]["XCenter"]
        int yCent = cellArr[3]["YCenter"]


    cdef np.ndarray[DTYPE_t, ndim=2] initCellPop = np.zeros((xMax, yMax), dtype=DTYPE)

    totCells = cellsXLattice * cellsYLattice

    xC = xCent - ((cellsXLattice-1)*gap)//2 - (cellsXLattice * lengthCellX)//2
    yC = yCent - ((cellsYLattice-1)*gap)//2 - (cellsYLattice * lengthCellY)//2


    cdef np.ndarray[DTYPE_t, ndim=2] cellP = np.arange(1, totCells+1, dtype=DTYPE).reshape(cellsXLattice, cellsYLattice)

    cellP = np.kron(cellP, np.ones((lengthCellX, lengthCellY), dtype=DTYPE))

    #--- Inserts gaps ---#
    if gap > 0:
        cellP = np.insert(cellP, gap*[i for i in range(lengthCellX, cellP.shape[0], lengthCellX)], 0, axis=0)
        cellP = np.insert(cellP, gap*[j for j in range(lengthCellY, cellP.shape[1], lengthCellY)], 0, axis=1)
    elif gap == 0:
        pass

    columnlength = cellP.shape[0]
    rowlength = cellP.shape[1]

    initCellPop[0:columnlength, 0:rowlength] = cellP

    initCellPop = np.roll(initCellPop, xC, axis=0)
    initCellPop = np.roll(initCellPop, yC, axis=1)

    return initCellPop
##################################################



########################################
###===== Find Nearest Neighbors =====###
########################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef list nearestneighbors((int, int) targetpx, int xMax, int yMax, int BC):
    #--- Statically type indexing variables ---#
    cdef Py_ssize_t r, i, j, k, s, t, m

    cdef:
        int x[9]
        int y[9]
        int m1[3]
        int m2[3]
    
    cdef:
        Py_ssize_t lenx = 9
        Py_ssize_t leny = 9
        list res
    
    #--- This is equivalent to np.arange ---#
    for t in range(-1, 2, 1):
        m1[t+1] = targetpx[0] + t
        m2[t+1] = targetpx[1] + t


    #--- This is equivalent to np.reshape ---#
    for r in range(9):
        y[r] = m2[r%3]
        with cython.cdivision(True):
            x[r] = m1[(r%9)//3]
 

    if BC == 1: #-- Periodic BC --#
        for i in range(lenx):
            if x[i] > xMax - 1:
                x[i] = 0
            elif x[i] < 0:
                x[i] = xMax - 1

        for j in range(leny):
            if y[j] > yMax - 1:
                y[j] = 0
            elif y[j] < 0:
                y[j] = yMax - 1

    if BC == 2: #-- Zero-Flux BC --#
        for k in range(lenx):
            if x[k] > xMax - 1:
                x[k] = xMax - 2
            elif x[k] < 0:
                x[k] = 1
    
        for s in range(leny):
            if y[s] > yMax - 1:
                y[s] = yMax - 2
            elif y[s] < 0:
                y[s] = 1

    res = x

    for m in y:
        res.append(m)

    return res
##################################################



##############################################
###===== Enhancement Pixel Connection =====###
##############################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int connectpixels((int, int) targetpx, np.ndarray[DTYPE_t, ndim=2] grid, int xMax, int yMax, int BC):

    cdef: 
        int xp[9]
        int yp[9]
        int x[8]
        int y[8]
        int val[8]
        int res[18]

    cdef:
        Py_ssize_t i, j, k, n, s
        Py_ssize_t lenz = 8
        Py_ssize_t lenx = 8 
        Py_ssize_t leny = 8 
        Py_ssize_t lenv = 8 
        int pcx = 0
        int indt, right, mid, left
        (int, int) alist[9]
        (int, int) clist[3]
        (int, int) dlist[5]
        (int, int) zlist[8]

    res = nearestneighbors(targetpx, xMax, yMax, BC) #-- list --#

    for i in range(9):
        xp[i] = res[i]
        yp[i] = res[i + 9]

    for j in range(9):
        alist[j] = (xp[j], yp[j])

    for k in range(3):
        clist[k] = alist[k]

    dlist = [alist[5], alist[8], alist[7], alist[6], alist[3]]

    zlist[:3] = clist
    zlist[3:8] = dlist

    for n in range(lenz):
        x[n] = zlist[n][0]
        y[n] = zlist[n][1]
    

    if BC == 1: #-- Periodic BC --#
        for i in range(lenx):
            if x[i] > xMax - 1:
                x[i] = 0
            elif x[i] < 0:
                x[i] = xMax - 1

        for j in range(leny):
            if y[j] > yMax-1:
                y[j] = 0
            elif y[j] < 0:
                y[j] = yMax - 1

    if BC == 2: #-- Zero-flux BC --#
        for k in range(lenx):
            if x[k] > xMax - 1:
                x[k] = xMax - 2
            elif x[k] < 0:
                x[k] = 1
    
        for s in range(leny):
            if y[s] > yMax - 1:
                y[s] = yMax - 2
            elif y[s] < 0:
                y[s] = 1


    #--- Target pixel in cell type index ---#
    indt = grid[targetpx[0], targetpx[1]]

    #--- Neighbor pixels in cell type index ---#
    for i in range(lenv):
        val[i] = grid[x[i], y[i]]

    for mid in range(lenv):
        left = mid - 1
        if left == -1:
            left = lenv - 1

        right = mid + 1
        if right == lenv:
            right = 0

        pcx += 1*(val[mid]==indt) * (2 - 1*(val[right]==indt) - 1*(val[left]==indt))

    return pcx
##################################################


#################################################################
###===== Calculate Cell Perimeter: Numpy's roll version =====###
#################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cellPerim(np.ndarray[DTYPE_t, ndim=2] grid):

    cdef:
        Py_ssize_t i, j 
        int vola, volb
        Py_ssize_t leng1 = grid.shape[0]
        Py_ssize_t leng2 = grid.shape[1]

    cdef:
        np.ndarray[DTYPE_t, ndim=2] a = np.empty((grid.shape[0], grid.shape[1]), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] b = np.empty((grid.shape[0], grid.shape[1]), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] cellBorder = grid

    a = 1*(cellBorder != np.roll(cellBorder,1,axis=1)) #-- horizontally --#
    b = 1*(cellBorder != np.roll(cellBorder,1,axis=0)) #-- vertically --#

    vola = 0
    for i in range(leng1):
        for j in range(leng2):
            vola += a[i, j]

    volb = 0
    for i in range(leng1):
        for j in range(leng2):
            volb += b[i, j]

    return vola + volb
##################################################


############################################
###===== Calculate Adhesion Energy =====###
############################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double adhesionEnergy((int, int) targetpx, np.ndarray[DTYPE_t, ndim=2] grid, list pottsElements, list contactEnergy):

    cdef:
        int xMax = pottsElements[0]["xDim"]
        int yMax = pottsElements[0]["yDim"]
        int BC = pottsElements[4]["Boundary"]
        double Jcm = contactEnergy[1]["Value"]
        double Jcc = contactEnergy[2]["Value"]

    cdef:
        int xp[9]
        int yp[9]
        int n1[9]
        int n2[9]
        int allneighbors[9]
        int res1[18]
        int res2[18]

    cdef:
        Py_ssize_t i, j, k, t, r
        (int, int) tpx 
        Py_ssize_t lenz = 9
    
    cdef:
        int kMed = 0, kCell1 = 0, kOthers = 0
        double ea = 0.0

    #---------------------------------#

    res1 = nearestneighbors(targetpx, xMax, yMax, BC)

    for i in range(9):
        xp[i] = res1[i]
        yp[i] = res1[i + 9]

    for j in range(lenz):
        tpx = (xp[j], yp[j])

        res2 = nearestneighbors(tpx, xMax, yMax, BC)
 
        for k in range(9):
            n1[k] = res2[k]
            n2[k] = res2[k + 9]

        for t in range(lenz):
            allneighbors[t] = grid[n1[t], n2[t]]


        kMed = 0
        kCell1 = 0
        kOthers = 0

        for r in range(lenz):
            if allneighbors[r] == 0: #-- Get sum of medium (index 0) --#
                kMed += 1
            elif allneighbors[r] == 1: #-- Get sum of cell (index 1) --#
                kCell1 += 1
            elif allneighbors[r] != 0 and allneighbors[r] != 1: #-- Get cell of other types
                kOthers += 1


        if grid[tpx[0], tpx[1]] == 0: #-- if target pixel is medium --#
            ea += <double> kCell1 * Jcm
        elif grid[tpx[0], tpx[1]] > 0: #-- otherwise, if target pixel is not medium --#
            ea +=  <double>kMed * Jcm
            ea += <double> kOthers * Jcc

    return ea
##################################################



########################################################
###===== Calculate Adhesion Energy due to CAMs =====###
########################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double adhesionFlex((int, int) targetpx, np.ndarray[DTYPE_t, ndim=2] grid, list pottsElements, list camElements):

    cdef:
        int xMax = pottsElements[0]["xDim"]
        int yMax = pottsElements[0]["yDim"]
        int BC = pottsElements[4]["Boundary"]

    cdef:
        double Ncol = camElements[0]["Density"]
        double Nint = camElements[1]["Density"]
        double Ncad = camElements[2]["Density"]
        double kcc = camElements[3]["Value"]
        double kcm = camElements[4]["Value"]

    cdef:
        double Jcc = kcc * fmin(Ncad, Ncad)
        double Jcm = kcm * fmin(Nint, Ncol)
        double JcmEnergy = 0.0
        double JccEnergy = 0.0

    cdef:
        int xp[9]
        int yp[9]
        int n1[9]
        int n2[9]
        int allneighbors[9]
        int res1[18]
        int res2[18]

    cdef:
        Py_ssize_t i, j, k, t, r, s
        (int, int) tpx 
        Py_ssize_t lenz = 9
        int step2, step5
        double step3, step6
        int idxt


    res1 = nearestneighbors(targetpx, xMax, yMax, BC)
    
    for i in range(9):
        xp[i] = res1[i]
        yp[i] = res1[i + 9]


    for j in range(lenz):
        tpx = (xp[j], yp[j])

        idxt = grid[tpx[0], tpx[1]]

        res2 = nearestneighbors(tpx, xMax, yMax, BC)

        for k in range(9):
            n1[k] = res2[k]
            n2[k] = res2[k + 9]
 
        for t in range(lenz):
            allneighbors[t] = grid[n1[t], n2[t]]


        step2 = 0
        step5 = 0

        if idxt == 0: # if target pixel is medium
            for r in range(lenz):
                if allneighbors[r] > 0: # if neighbors are not medium
                    step2 += 1

            step3 = <double>step2 * Jcm
            JcmEnergy += step3

        elif idxt > 0: # if target pixel is not medium
            for r in range(lenz):
                if allneighbors[r] == 0: # if neighbors are medium
                    step2 += 1

            step3 = <double>step2 * Jcm
            JcmEnergy += step3

            for s in range(lenz):
                if allneighbors[s] != 0 and allneighbors[s] != idxt:
                    step5 += 1

            step6 = <double>step5 * Jcc
            JccEnergy += step6


    return - (JcmEnergy + JccEnergy)
##################################################


##########################################
###===== Calculate Geometric Mean =====###
##########################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float geo_mean_overflow(list iterable):

    '''
    Calculates the geometric mean of a list of numbers/integers
    with overflow prevention
    '''

    cdef Py_ssize_t arrlength = len(iterable)
    cdef Py_ssize_t i, j
    cdef double[:] arr = np.empty((arrlength, ), dtype=np.double)
    cdef double s = 0.0, res
    
    #--- Map the numbers to a log domain ---#
    for j in range(arrlength):
        if iterable[j] == 0:
            arr[j] = 0.0
        else:
            arr[j] = log(<double>iterable[j])

    #--- Calculate the sum of these logs ---#
    for i in range(arrlength):
        s += <double>arr[i]

    #--- Multiply the sum by 1/arrlength and calculate the exponent --#
    if s == 0.0:
        res = 0.0
    else:
        with cython.cdivision(True):
            res = exp(s/arrlength)

    return res
##################################################



############################################################
###===== Implementation of the Actin/Membrane Model =====###
############################################################
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int gmfunc((int, int) targetpx, (int, int) sourcepx, np.ndarray[DTYPE_t, ndim=2] grid, np.ndarray[DTYPE_t, ndim=2] actin, list pottsElements):

    cdef:
        list ress, rest, gs3, gt3, gs4, gt4
        int gs2, gt2, v
        ((int, int), int) ds[9]
        ((int, int), int) dt[9]
        double gms, gmt
        int gm
        # tuple k
        int i1, j1, k1, l1, n1, m1, i2, j2, k2, m2, l2, n2
        int xs[9]
        int xt[9]
        int ys[9]
        int yt[9]
        int gs1[9]
        int gt1[9]
        (int, int) nns[9]
        (int, int) nnt[9]
        int lengs3, lengt3

    cdef:
        int xMax = pottsElements[0]["xDim"]
        int yMax = pottsElements[0]["yDim"]
        int BC = pottsElements[4]["Boundary"]


    #===== (1) Use the source pixel =====#
    #--- Get Moore neighbors of the source pixel ---#
    ress = nearestneighbors(sourcepx, xMax, yMax, BC)

    for i1 in range(9):
        xs[i1] = ress[i1] #--- C array ---#
        ys[i1] = ress[i1 + 9] #--- C array ---#

    #--- List of tuples containing the Moore neighbors ---#
    for j1 in range(9):
        nns[j1] = (xs[j1], ys[j1]) #--- C array of tuples ---#

    #--- The neighbors in cell type index ---#
    for k1 in range(9):
        gs1[k1] = grid[xs[k1], ys[k1]] 

    #--- The source pixel in cell type index ---#
    gs2 = grid[sourcepx[0], sourcepx[1]] #--- int ---#

    #--- Find neighbors that have the same index as the source pixel ---#
    for l1 in range(9):
        ds[l1] = (nns[l1], gs1[l1])

    gs3 = []
    for n1 in range(9):
        if ds[n1][1] == gs2:
            gs3.append(ds[n1][0])

    lengs3 = len(gs3)

    gs4 = []
    for m1 in range(lengs3):
        gs4.append(actin[gs3[m1]]) #--- list ---#
    
    gms = geo_mean_overflow(gs4)


    #===== (2) Use the target pixel =====#
    #--- Get Moore neighbors of the target pixel ---#
    rest = nearestneighbors(targetpx, xMax, yMax, BC)

    for i2 in range(9):
        xt[i2] = rest[i2] #--- C array ---#
        yt[i2] = rest[i2 + 9] #--- C array ---#

    #--- List of tuples containing the Moore neighbors ---#
    for j2 in range(9):
        nnt[j2] = (xt[j2], yt[j2]) #--- C array of tuples ---#

    #--- The neighbors in cell type index ---#
    for k2 in range(9):
        gt1[k2] = grid[xt[k2], yt[k2]]

    #--- The target pixel in cell type index ---#
    gt2 = grid[targetpx[0], targetpx[1]] #--- int ---#

    #--- Find neighbors that have the same index as the source pixel ---#
    for l2 in range(9):
        dt[l2] = (nnt[l2], gt1[l2])

    gt3 = []
    for n2 in range(9):
        if dt[n2][1] == gt2:
            gt3.append(dt[n2][0])

    lengt3 = len(gt3)

    gt4 = []
    for m2 in range(lengt3):
        gt4.append(actin[gt3[m2]]) #--- list ---#
    
    gmt = geo_mean_overflow(gt4)

    gm = <int>round(gms) - <int>round(gmt)

    return gm
##################################################


###################################################
###===== ODE Solver: 4th-Order Runge-Kutta =====###
###################################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef list RK4thOrder(f, np.ndarray[double, ndim=1] yinit, double xinit, double h):
    cdef: 
        int m = len(yinit)
        double x = xinit
        np.ndarray[double, ndim=1] y = yinit
        Py_ssize_t j

        
    cdef:
        double c2 = 1.0/2.0 
        double c3 = 1.0/2.0 
        double c4 = 1.0
        double a21 = 1.0/2.0
        double a31 = 0.0
        double a32 = 1.0/2.0
        double a41 = 0.0 
        double a42 = 0.0 
        double a43 = 1.0/2.0
        double b1 = 1.0/6.0 
        double b2 = 1.0/3.0 
        double b3 = 1.0/3.0 
        double b4 = 1.0/6.0 

    cdef:
        np.ndarray[double, ndim=1] k1 = np.zeros((2, ), dtype=np.double)
        np.ndarray[double, ndim=1] k2 = np.zeros((2, ), dtype=np.double)
        np.ndarray[double, ndim=1] k3 = np.zeros((2, ), dtype=np.double)
        np.ndarray[double, ndim=1] k4 = np.zeros((2, ), dtype=np.double)

    
    k1 = h * f(x, y)

    k2 = h * f(x+c2*h, y + a21*k1)
 
    k3 = h * f(x+c3*h, y + a31*k1 + a32*k2)

    k4 = h * f(x+c4*h, y + a41*k1 + a42*k2 + a43*k3)

    for j in range(m):
        y[j] = y[j] + b1*k1[j] + b2*k2[j] + b3*k3[j] + b4*k4[j]

    x = x + h

    return [x, y]
###################################################



#######################################################
###===== The Intracellular Dynamics of MT1-MMP =====###
#######################################################
def myFunc(x, y):
    #-- ODEs
    ks = 0.1
    k1 = 0.16 
    k2 = 0.022
    k3 = 0.01
    m = 2
    Km = 2
    Ks = 1
    a = 0.0
       
    dy = np.zeros((len(y)))

    dy[0] = ks - k1*y[0]**m/(1+y[0]**m)
    dy[1] = k3*y[1] + k1*y[0]**m/(Km+y[0]**m) - k2*y[1]/(Ks+a*y[1])

    return dy



###################################
###===== The Main Function =====###
###################################
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cpm(list pottsElements, list cellArr, list contactEnergy, list camElements, list volumeElements, list surfaceElements, list actinElements, saveImages):

    ###===== Initialization for plotting =====###
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #############################################


    ###===== Potts elements =====###
    cdef:
        int xMax = pottsElements[0]["xDim"]
        int yMax = pottsElements[0]["yDim"]        
        Py_ssize_t mcs = pottsElements[1]["Steps"]
        int T = pottsElements[2]["Temperature"]
        int BC = pottsElements[4]["Boundary"]
    ################################


    ###===== Contact energy elements =====###
    cdef:
        double Jmm = contactEnergy[0]["Value"]
        double Jcm = contactEnergy[1]["Value"]
        double Jcc = contactEnergy[2]["Value"]
    #########################################


    ###===== Cell adhesion molecule elements =====###
    cdef:
        double Ncol = camElements[0]["Density"]
        double Nint = camElements[1]["Density"]
        double Ncad = camElements[2]["Density"]
        double kcc = camElements[3]["Value"]
        double kcm = camElements[4]["Value"]
    #################################################


    ###===== Volume constraint elements =====###
    cdef:
        int volTarget = volumeElements[0]["TargetVolume"]
        int volPlugin = volumeElements[1]["PluginType"]
        double circ = volumeElements[2]["Circularity"]
        double lambdaVol = volumeElements[0]["LambdaVolume"]
    ############################################


    ###===== Surface constraint elements =====###
    cdef double lambdaSurf = surfaceElements[0]["LambdaSurface"]
    #############################################


    ###===== Actin elements =====###
    cdef:
        int membranemt1 = actinElements[0]["MembraneMT1"]
        double lambdaM = actinElements[0]["LambdaMT1"]
    ################################

    
    ###===== Initialization for nearestneighbors =====###
    cdef:
        int z1[9]
        int z2[9]
        Py_ssize_t lenz = 9
        Py_ssize_t k, r, u, v
        int res[18]
        (int, int) zz[9]
    #####################################################


    ###===== Monte Carlo Steps =====###
    cdef:
        Py_ssize_t noLattice = xMax * yMax
        Py_ssize_t i, j
    ###################################


    ###===== Initialization for cell volume and area =====###
    cdef:
        double HSurf, HSurfNew, HVol, HVolNew, HSize, HSizeNew
        double deltaHSize
        int cellVol=0, cellVolNew, cellArea=0, cellAreaNew
        int Vt = 0
        double St = 0.0
        Py_ssize_t s, t

    #--- Calculate target volume ---#
    if volPlugin == 1: # VolumePlugin
        Vt = volTarget 
    elif volPlugin == 2: # VolumeLocalFlex
        Vt = volTarget

    #--- Calculate target surface using circularity ---#
    St = sqrt(4 * M_PI * Vt / circ)
    #########################################################


    ###===== Initialization for cell adhesion =====###
    cdef:
        double HAdhesion, HAdhesionNew, deltaHAdhesion
        double HFlex, HFlexNew, deltaHFlex
    ##################################################


    ###===== Initialization for MT1-MMP dynamics =====###        

    cdef:
        double h=0.1, x_init=0.0, xinit, ts, alpha, activemt1
        int nsteps = 10 
        double deltaHActin
        Py_ssize_t ii
        
    cdef:
        np.ndarray[double, ndim=1] tsol = np.zeros((nsteps,), dtype=np.double)
        np.ndarray[double, ndim=1] ysol1 = np.zeros((nsteps,), dtype=np.double)
        np.ndarray[double, ndim=1] ysol2 = np.zeros((nsteps,), dtype=np.double)
        np.ndarray[double, ndim=1] y_init = np.array([5.5, 0.0], dtype=np.double)
        np.ndarray[double, ndim=1] ys = np.zeros((2,), dtype=np.double)
        np.ndarray[double, ndim=1] tt = np.zeros((mcs+2,), dtype=np.double)
        np.ndarray[double, ndim=1] mmp1 = np.zeros((mcs+2,), dtype=np.double)
        np.ndarray[double, ndim=1] mmp2 = np.zeros((mcs+2,), dtype=np.double)

    tsol[0] = x_init
    ysol1[0] = y_init[0]
    ysol2[0] = y_init[1]

    tt[0] = x_init
    mmp1[0] = y_init[0]
    mmp2[0] = y_init[1]
    #####################################################
     

    ###===== Index copy attempt properties =====## 
    cdef:
        unsigned int idxt, idxs
        int rc=0
        (int, int) targetpx, sourcepx     
        double deltaH=0
        double prob, k1, Integrin
    ##############################################

            
    ###===== Generate all grids =====##
    cdef:
        np.ndarray[DTYPE_t, ndim=2] grid = initializePopulation(pottsElements, cellArr)
        Py_ssize_t leng1 = grid.shape[0]
        Py_ssize_t leng2 = grid.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] newgrid = np.empty((leng1, leng2), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=2] act = np.zeros((leng1, leng2), dtype=DTYPE)

    act = (grid>0) * membranemt1
    ###################################
    
    cdef:
        tuple pos
        double start, stop
    
    #-- Uncomment to print the center of mass --#
    # posfile = open('Results/center_of_mass.txt', 'w')

    cdef str saveFigures = saveImages[0]["SaveImages"]
    cdef int save_each

    srand(np.random.randint(1,8000000))
    # np.random.seed(1234) #-- Uncomment to reproduce similar results --#

    start = time.time()

    for j in xrange(0, (mcs+1)):

        for i in xrange(0, (noLattice+1)):

            #-- Randomly selects a target pixel --#
            targetpx = (<int>(rand()/<double>RAND_MAX*xMax), <int>(rand()/<double>RAND_MAX*yMax))

            #-- The target pixel in cell type index --#
            idxt = grid[targetpx[0], targetpx[1]]

            #-- Generates a list of Moore neighbors --#
            res = nearestneighbors(targetpx, xMax, yMax, BC)

            for k in range(lenz):
                z1[k] = res[k]
                z2[k] = res[k + lenz]

            #-- Moore neighbors in Python coordinate --#
            for r in range(lenz):
                zz[r] = (z1[r], z2[r])

            #-- Randomly selects a source pixel from the Moore neighbors --#
            for _ in range(lenz):
                rc = int(rand()/<double>RAND_MAX*lenz)

            sourcepx = zz[rc]

            #-- The source pixel in cell type index --#
            idxs = grid[sourcepx[0], sourcepx[1]]


            if idxt != idxs: #-- if target pixel not equal source pixel --#

                ###===== (1) Size (volume + area) Constraints =====###
                #-------------------------------#
                #-- Calculate the actual volume --#
                cellVol = 0
                for s in range(leng1):
                    for t in range(leng2):
                        cellVol += grid[s, t]

                #--- Calculate the actual surface ---#
                cellArea = cellPerim(grid)

                #-- Hamiltonian of Volume & Surface Constraints ---#
                HSurf = lambdaSurf * (cellArea - St)**2
                HVol = lambdaVol * (cellVol - Vt)**2

                HSize = HSurf + HVol

                #-- Create a new grid --#
                newgrid = copy.deepcopy(grid)

                #-- The index copy attempt --#
                #-- The source pixel occupies the target pixel --#
                newgrid[targetpx[0], targetpx[1]] = idxs

                #-- New cell volume after the index copy attempt --#
                if idxs == 0: #-- if the source pixel is medium --#
                    cellVolNew = cellVol - 1 #-- Cell volume decreases due to the index copy attempt --#
                else:         #-- if the source pixel is NOT medium --#
                    cellVolNew = cellVol + 1 # Cell volume increases due to the index copy attempt --#

                #-- New cell area after the index copy attempt --#
                cellAreaNew = cellPerim(newgrid) 

                #-- New size contraints --#
                HSurfNew = lambdaSurf * (cellAreaNew - St)**2
                HVolNew = lambdaVol * (cellVolNew - Vt)**2

                HSizeNew = HSurfNew + HVolNew

                deltaHSize = HSizeNew - HSize

                ###===== (2) Adhesion (Cell-Cell + Cell-Matrix) Constraints =====###
                if volPlugin == 1:
                    ###===== Regular Adhesion Energy =====###
                    HAdhesion = adhesionEnergy(targetpx, grid, pottsElements, contactEnergy)
                    HAdhesionNew = adhesionEnergy(targetpx, newgrid, pottsElements, contactEnergy)
                    deltaHAdhesion = HAdhesionNew - HAdhesion
                    deltaH = deltaHSize + deltaHAdhesion
                elif volPlugin == 2:
                    ##===== Adhesion Flex Energy =====###
                    HFlex = adhesionFlex(targetpx, grid, pottsElements, camElements)
                    HFlexNew = adhesionFlex(targetpx, newgrid, pottsElements, camElements)
                    deltaHFlex = HFlexNew - HFlex         
                    deltaH = deltaHSize + deltaHFlex

                ##===== Actin Model =====###
                deltaHActin = (lambdaM/membranemt1)*gmfunc(targetpx, sourcepx, grid, act, pottsElements)
                '''
                The success probability of the copy attempt is biased by subtracting
                deltaHActin from deltaH:
                '''
                deltaH = deltaH - deltaHActin

                if connectpixels(targetpx, grid, xMax, yMax, BC) > 2:
                    deltaH += 10000000.0


                ###===== Without the Membrane Model =====###
                # if exp(-deltaH / T) * RAND_MAX > rand():
                #     grid = newgrid
                #     cellArea = cellAreaNew
                #     cellVol = cellVolNew
                #     HSize = HSizeNew
                    
                ###===== Including the Membrane Model =====###
                '''
                The empty lattice sites that form the medium have a zero activity
                value, while sites that are freshly incorporated by a cell get
                the maximum activity value (MaxAct)
                '''
                if exp(-deltaH / T) * RAND_MAX > rand():
                    if idxs == 0: #-- If the source pixel is medium --#
                        act[targetpx[0], targetpx[1]] = 0 #-- retract membrane --#
                    elif idxs == 1: #-- If the source pixel is cell --#
                        act[targetpx[0], targetpx[1]] = membranemt1 #-- expand membrane --#


                    #--- Update the State ---#
                    grid = newgrid
                    cellArea = cellAreaNew
                    cellVol = cellVolNew
                    HSize = HSizeNew

                    
            if i == noLattice:

                ##--- Print center of mass position ---##
                # pos = ndimage.measurements.center_of_mass(grid)
                # posfile.write("%s %s\n" % (j, str(pos)))


                ###===== Print/plot the solution =====###
                ##--- Use these when including the Membran Model
                ax.clear()
                
                # ima=ax.matshow(grid, cmap='Greys', interpolation='none', origin='lower', vmin=0, vmax=1.5)
                # ima=ax.matshow(act, interpolation='nearest', cmap='terrain', origin='lower', alpha=0.4, vmin=0, vmax=50)
                
                ima=ax.matshow(grid, cmap='Purples', interpolation='none', origin='lower', vmin=0.0, vmax=grid.max())
                ima=ax.matshow(act, interpolation='nearest', cmap='BuPu', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())

                # ima=ax.matshow(act, interpolation='nearest', cmap='Blues', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='BuGn', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())               
                # ima=ax.matshow(act, interpolation='nearest', cmap='GnBu', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='Greens', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='Greys', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='Oranges', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='OrRd', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='PuBu', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='PuBuGn', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='PuRd', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='RdPu', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='Reds', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='YlGn', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='YlGnBu', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='YlOrBr', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                # ima=ax.matshow(act, interpolation='nearest', cmap='YlOrRd', origin='lower', alpha=0.5, vmin=0.0, vmax=act.max())
                
                ax.set_axis_off()
                
                ##--- To show colorbar of MT1-MMP --##
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.05)
                # plt.colorbar(ima, cax=cax)

                ###===== Without the Actin Model =====###
                # ax.clear()
                # ax.matshow(grid, cmap=plt.cm.afmhot, interpolation='none', vmin=0, vmax=2.1)

                # fig.suptitle(f'MCS = {j}')
                plt.pause(0.00001) 
                #########################################


                ###===== Intracellular MT1-MMP =====###
                for ii in range(nsteps):
                    ts, ys = RK4thOrder(f=myFunc, yinit=y_init, xinit=x_init, h=h)
                    tsol[ii] = ts
                    ysol1[ii] = ys[0]
                    ysol2[ii] = ys[1]
                    x_init = ts
                    y_init = np.array([ys[0], ys[1]])

                x_init = tsol[nsteps-1]
                y_init = np.array([ysol1[nsteps-1], ysol2[nsteps-1]])

                mmp2[j+1] = ysol2[nsteps-1]

                activemt1 = mmp2[j]


                ###===== Membrane MT1-MMP =====###
                membranemt1 = actinElements[0]["MembraneMT1"]
                alpha = 2.675 #-- for Max MT1MMP = 16; 
                membranemt1 = int(alpha * activemt1)
                actinElements[0]["MembraneMT1"] = membranemt1

                ###===== Internalization =====###
                ##--- The activity value of a site decreases by one after every MCS --#
                act[act!=0] = act[act!=0] - 1


        print(f'MCS = {j}, membrane MT1-MMP = {membranemt1}')

        save_each = 10 #-- Save images every 10 MCS --#

        if saveFigures == 'Yes':
            if j % save_each == 0:
                printImages(j)
        elif saveFigures == 'No':
            pass
        

             
    #-- Uncomment these for displaying images --#
    plt.show(block=False) 
    plt.pause(0.01) 
    plt.close()
    
    #-- Comment out above and uncomment this for no image display --#
    # plt.close(fig) #-- for no display --#

    stop = time.time()

    print('Simulation finished, t = %.2f sec' % (stop-start))

    # posfile.close()
    




    