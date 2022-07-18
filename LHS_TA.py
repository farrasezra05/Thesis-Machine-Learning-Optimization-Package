import numpy as np
from math import factorial
import xlsxwriter
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from random import randrange
import random as rn
from IPython.display import display
import pandas as pd
import math

rn.seed(0)

os.chdir(r"C:\Amanah\1-AE\2-BISMILLAH TA\1-TA Ezra\Python\ANN")

__all__ = ['lhs']

def lhs(n, samples=None, criterion=None, iterations=None):
    H = None

    if samples is None:
        samples = n
    if criterion is not None:
        assert criterion.lower() in ('center', 'c', 'maximin', 'm','centermaximin', 'cm', 'centermaximin1'), 'Invalid value for "criterion": {}'.format(criterion)
    if criterion is None:
        criterion = 'center'
    if iterations is None:
        iterations = 5
    if H is None:
        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, samples)
        elif criterion.lower() in ('maximin','m'):
            H = _lhsmaximin(n, samples, iterations, 'maximin')
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, samples, iterations, 'centermaximin')
        elif criterion.lower() in ('centermaximin1', 'cm'):
            H = _lhsmaximin1(n, samples, iterations, 'centermaximin1')
    return H

def _lhscentered(n, samples):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)
    u = np.random.rand(samples + 1, n)

    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = np.random.permutation(cut)

    return H
def _lhscentered1(n, samples):
    # Generate the intervals
    cut = np.linspace(0.1, 1, samples)
    u = np.random.rand(samples, n)

    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = np.random.permutation(cut)
    return H

################################################################################
def _lhsmaximin(n, samples, iterations,lhstype):
    maxdist = 0
    # Maximize the minimum distance between points
    for i in range(iterations):
        Hcandidate = _lhscentered(n,samples)
        d = _pdist(Hcandidate)
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    return H

def _lhsmaximin1(n, samples, iterations, lhstype):
    maxdist = 0
    # Maximize the minimum distance between points
    for i in range(iterations):
        Hcandidate = _lhscentered1(n, samples)
    d = _pdist(Hcandidate)
    if maxdist < np.min(d):
        maxdist = np.min(d)
        H = Hcandidate.copy()
    return H

################################################################################
def _pdist(x):
    x = np.atleast_2d(x)
    assert len(x.shape) == 2

    m, n = x.shape
    if m < 2:
        return []

    d = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            d.append((sum((x[j, :] - x[i,:]) ** 2)) ** 0.5)
    return np.array(d)

def round_up_to_even(f):
    return np.ceil(f / 2.) * 2

rn.seed(0)
dimensi = lhs(4, samples=10, criterion='centermaximin1')
orientasi = lhs(6, samples=12, criterion='centermaximin')
cross = randrange(1, 5) #variate cross section
mat = randrange(1,5) #variate material
x = 0
for x in range(9):
    cross = np.vstack([cross, randrange(1, 5)])
    mat = np.vstack([mat, randrange(1, 5)])
    x = x + 1

y = 0
for y in range(9):
    dimensi2 = lhs(4, samples=10, criterion='centermaximin')
    dimensi = np.vstack([dimensi, dimensi2])
    orientasi2=lhs(6, samples=12, criterion='centermaximin')
    orientasi=np.vstack([orientasi,orientasi2])
    cross2=randrange(1,5)
    mat2=randrange(1,5)
    x=0
    for x in range (9):
        cross2=np.vstack([cross2,randrange(1, 5)])
        mat2 = np.vstack([mat2, randrange(1, 5)])
        x=x+1
    cross =np.vstack([cross,cross2])
    mat = np.vstack([mat, mat2])
    y=y+1

dimensi[:,0]=4.5+(1.5*dimensi[:,0]) #gaplength, 8-15 mm
dimensi[:,1]=1+2*dimensi[:,1] #thickness, 1-3 mm
dimensi[:,2]=45+np.floor(15*dimensi[:,2]) #angle corner, 45 deg
dimensi[:,3]=2+round_up_to_even(10*dimensi[:,3]) #layerkomposit, 2-12
display(dimensi)
display(orientasi)
display(cross)

i=0
if dimensi[0,1]!=None:
    array=np.hstack([dimensi[i,:],180*orientasi[i,:]])
    i=1
    for i in range(100):
        if dimensi[i,1]!=None:
            array1=np.hstack([dimensi[i,:],180*orientasi[i,:]])
            array=np.vstack([array,array1])
        i=i+1
i=0

#DELETING LAMINATE ORIENTATION IN NON-EXIST LAYERS
for i in range(100):
    j=0
    for j in range(7):
        if j > (array[i,3]/2):
            array[i,3+j]=0
        j=j+1
    i=i+1

print(mat.shape)
print(array.shape)
array=np.delete(array, 0, axis=0)
array=np.hstack([array,cross])
array=np.hstack([array,mat])

#DELETING LAMINATE & ITS ORIENTATION IN NON-COMPOSITE MODEL
for i in range(100):
    if ((array[i, 11] == 3) or (array[i, 11] == 4)):
        array[i, 3] = 0
    i=i+1
i=0
for i in range(100):
    j=0
    for j in range(7):
        if j > (array[i,3]/2):
            array[i,3+j]=0
        j=j+1
    i=i+1
i=0
for i in range(100):
    if (array[i,1])<0.1:
        array = np.delete(array, i, axis=0)
        i = i - 1
    i = i + 1

title = [["Side Length"],
         ["Thickness"],
         ["Cell Angle"],
         ["#layers"],
         ["Ori.1,12"],
         ["Ori.2,11"],
         ["Ori.3,10"],
         ["Ori.4,9"],
         ["Ori.5,8"],
         ["Ori.6,7"],
         ["X-Sect"],
         ["Mat"]]
arraytranspose=array.transpose()
data = arraytranspose[10,:]
data2 = arraytranspose[11,:]
arraytranspose = np.hstack((title, arraytranspose))
bentuk=[]
material=[]

for i in range(100):
    if data[i]==1:
        bentuk.append('Re-entrant')
    if data[i]==2:
        bentuk.append('DAH')
    if data[i]==3:
        bentuk.append('Star')
    if data[i]==4:
        bentuk.append('DUH')
    i=i+1
for i in range(100):
    if data2[i]==1:
        bentuk.append('GFRP')
    if data2[i]==2:
        bentuk.append('CFRP')
    if data2[i]==3:
        bentuk.append('Carbon Steel')
    if data2[i]==4:
        bentuk.append('Aluminum')
    i=i+1

re_count = bentuk.count('Re-entrant')
dah_count = bentuk.count('DAH')
star_count = bentuk.count('Star')
duh_count = bentuk.count('DUH')
gfrp_count = bentuk.count('GFRP')
cfrp_count = bentuk.count('CFRP')
carbst_count = bentuk.count('Carbon Steel')
alu_count = bentuk.count('Aluminum')
display(re_count)
display(dah_count)
display(star_count)
display(duh_count)
display(gfrp_count)
display(cfrp_count)
display(carbst_count)
display(alu_count)
shape=['Re-entrant','DAH','Star','DUH']
bahan=['GFRP','CFRP','Carbon Steel','Aluminum']
count=[re_count, dah_count, star_count, duh_count]
count2=[cfrp_count, gfrp_count, carbst_count, alu_count]

display(shape)
display(count)
display(bahan)
display(count2)

display(array)

from matplotlib.ticker import AutoMinorLocator

plt.figure(1)
fig, ax = plt.subplots(2, 2, figsize=[10,15])
print(ax)
plt.minorticks_on()
ax[0,0].scatter(x=array[:,0], y=array[:,1])
ax[0,0].set_xlabel("Cell Side Length (mm)")
ax[0,0].set_ylabel("Material Thickness (mm)")

ax[1,0].scatter(x=array[:,1], y=array[:,2])
ax[1,0].set_xlabel("Cell Thickness (mm)")
ax[1,0].set_ylabel("Cell Corner Angle (deg)")

ax[0,1].scatter(x=array[:,0], y=array[:,2])
ax[0,1].set_xlabel("Cell Inner Spacing (mm)")
ax[0,1].set_ylabel("Cell Corner Angle (deg)")

ax[1,1].scatter(x=array[:,3], y=array[:,4])
ax[1,1].set_xlabel("Composite Layers (mm)")
ax[1,1].set_ylabel("Layer 1 Orientation (deg)")

plt.figure(2)
fig, ax = plt.subplots(1, 2, figsize=[10,15])
ax[0].bar(shape, count)
ax[0].set_xlabel("Cell Cross Section")
ax[0].set_ylabel("Value")

ax[1].bar(bahan, count2)
ax[1].set_xlabel("Material")
ax[1].set_ylabel("Value")

plt.show()

workbook = xlsxwriter.Workbook('arrays.xlsx',  {'strings_to_numbers': True})
worksheet = workbook.add_worksheet()
row = 0

print('arraytranspose')
print(arraytranspose)

for col, data in enumerate(arraytranspose):
    worksheet.write_column(row, col, data)

workbook.close()

fig = plt.figure(2, figsize=(8, 6), dpi=80)
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = array[:,0]
    ys = array[:,1]
    zs = array[:,2]

ax.scatter(xs, ys, zs, c=c, marker=m)
ax.set_xlabel('Cell Inner Spacing (h)')
ax.set_ylabel('Material Thickness')
ax.set_zlabel('Cell Corner Angle (teta)')

plt.show()