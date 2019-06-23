
################### WFGS Slit Finder ##############################3
##### Initial Version at github:
#####
##### 2019/03/19   --- Stefan Baar  ---- sbaar@nhao.jp
##### Change Log:
#####   DATE       ---    NAME      ----    contact
#####
##### If possible, please fork changes at github.
#####
#####
##### finds slit and writes region file
#####
#####
import sys

import numpy as np
import cv2

from astropy.io import fits

from scipy import ndimage
from scipy.signal import find_peaks


#### help
def print_help():
        print "-------------------------------------"
        print "This script only takes one argument, which is a fits file containing a 2d data array"
        print ""
        print "Usage: slitfingeder [FILE.fits]"
        print ""
        return 0

if len(sys.argv) == 1:
    print_help()
    sys.exit()

elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
    print_help()
    sys.exit()
else:
    pass
#### normalize the image either to 8bit or 16bit
def normalize(IMAGE, BIT = 8):
    IMAGE0   = IMAGE-IMAGE.min()
    IMAGE1   = IMAGE0/IMAGE0.max()*2**int(BIT)
    return IMAGE1.astype("uint"+str(int(BIT)))

#### tries to remove the background and enhence the slit profile
def enhenceCV2(IM):
    IM0 = cv2.blur(IM,(50,50)).astype("float")
    IM1 = normalize(IM.astype("float")  - IM0)
    IM2 = cv2.blur(IM1,(40,40)).astype("float")
    return IM2

#### fits data points to the image edges
def linear_fit(y,x, image):
    COEF = np.polyfit(x,y,1)
    x1  = 0.
    x2  = image.shape[0]
    y1 = COEF[0]*x1+COEF[1]
    y2 = COEF[0]*x2+COEF[1]
    return np.asarray([y1,y2]),np.asarray([x1,x2]), COEF

#### resolves the transition area between the thick and the slim slit
#### by averaging the image along the x axis
def get_slit_peaks(SLIT, distance = 500):
    y , _ = find_peaks(SLIT.mean(1), distance=distance)
    x     = SLIT.mean(1)[y]
    SMASK = np.argsort(x)[-2:]
    x,y   = x[SMASK],sorted(y[SMASK])
    x     = np.argmax(SLIT[y[0]:y[1]].mean(0))
    x     = np.array([x,x])
    return np.array([x,y])

#### Rotates the detected slit vectors
def vector_rotate(origin, point, angle):
    angle = angle/180.*np.pi
    oy, ox = float(origin[0])      , float(origin[1])
    py, px = point[0].astype(float), point[1].astype(float)

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qy, qx

#### sorting
def sort_xy(sorting,tosort):
    sort_mask = np.argsort(sorting)
    tosort    = tosort[sort_mask]
    sorting   = sorting[sort_mask]
    return sorting,tosort

crop = 500

if len(sys.argv) < 2:
    IN = "slv190202_test.fits"
else:
    IN = sys.argv[1]

OUT= IN[:-5]+".reg"

###### Image enhencement
IMAGE               = np.asarray(fits.open(IN)[0].data).astype(float)
IMAGE_enhenced      = enhenceCV2(IMAGE)
IMAGE_threshold     = cv2.threshold(normalize(IMAGE_enhenced),100,100,cv2.THRESH_BINARY_INV)[1]
IMAGE_canny         = cv2.Canny(IMAGE_threshold ,1,1, apertureSize = 7)

###### get slit contours and resolve as 2D point List
y ,x  = np.where(IMAGE_canny[crop:-crop,crop:-crop] == 255)
x += crop
y += crop

###### fit slit contours
Xfit, Yfit, ANG = linear_fit(x,y,IMAGE_threshold)

XY_sort_mask    = np.argsort(Xfit)
Yfit            = Yfit[XY_sort_mask]
Xfit            = Xfit[XY_sort_mask]

###### Derotate the image to resolve whide slit position
origin   = (IMAGE_threshold.shape[1]/2,IMAGE_threshold.shape[0]/2)
rotation = 180.-np.arctan(ANG[0])/np.pi*180.
if rotation > 90:
        rotation = rotation - 180.
print("Slit Rotation Angle: "+str(round(rotation,2))+"deg")

###### create rotation matrix
rot_mat = cv2.getRotationMatrix2D(origin,rotation,1.)
rotANG  = cv2.warpAffine(IMAGE_threshold , rot_mat,
                             (IMAGE_threshold .shape[1],IMAGE_threshold .shape[0]),
                             flags=cv2.INTER_LINEAR)
x, y    = get_slit_peaks(rotANG)
X, Y    = vector_rotate(origin, (x,y), -rotation)
X, Y     = sort_xy(X, Y )

FILE = open(OUT, "w")
FILE.write("global color=pink\n")   #### Yes, the color has to be PINK at any time!!!
FILE.write("line "+str(Xfit[0])+" "+str(Yfit[0])+" "+str(X[0])+" "+str(Y[0])+"\n")
#FILE.write("line "+str(X[0])   +" "+str(Y[0])   +" "+str(X[1])+" "+str(Y[1])+"\n")  ### center line
FILE.write("line "+str(X[1])   +" "+str(Y[1])   +" "+str(Xfit[1])+" "+str(Yfit[1]))
FILE.close()
