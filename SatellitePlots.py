#======================================================================================
""" Importing the necessary Python libraries 
"""

import sys
sys.path.append('/Users/ayushdhananjai/Documents/UndergraduateResearch/HostPlots.py')
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

#======================================================================================
""" Reading the file, looping over the lines and defining 
    the required arrays 
"""

# Reading in the file line by line 

f = open('HostsSats.txt', 'r')
lines = f.read().split(',')
f.close()
line = lines[0].split('\n')

for j in range(19023):
    line[j] = line[j].strip()

j = 0
SatelliteGalaxies = []


# Looping over all satellites

for i in range(19023):
    if float(line[i][0:5]) < 10000: 
        # In the text file, only the host galaxy 
        # lines are indented and the ID number of the host is always 
        # greater than 10000. So anything else is a satellite.                   
        line[i] = line[i].split(' ')
        good = [len(x) > 0 for x in line[i]]
        line[i] = np.array(line[i])
        SatelliteGalaxies.append(line[i][good])
   
SatelliteGalaxies = np.array(SatelliteGalaxies[:]).astype('float')

# Calling the desired columns from the file

SatelliteStellarMass = SatelliteGalaxies[:,5]
SatelliteFluxRatio = SatelliteGalaxies[:,4]
SatelliteRmag = SatelliteGalaxies[:,3]
radius3d = SatelliteGalaxies[:,6]
radius3d.sort()

intervals = [0.00143353024, 0.381622791, 0.640596211, 0.873146892, 1.12500513,
             1.39045358, 1.67548299, 1.99902666]

#------------------------------------------------------------------------------------------------------------------

# FIRST RADIUS INTERVAL

deltaVx = []
deltaVy = []
deltaVz = []

count = 0

# Calculating deltaVx, deltaVy and deltaVz

for n, i in enumerate(HostGalaxies[:,13].astype('int')):
    assoc = SatelliteGalaxies[count:count+i,:]   
    interval = (assoc[:,6] >= intervals[0]) & (assoc[:,6] <= intervals[1])
    deltaVx.append(HostGalaxies[n,6] - (assoc[:,7][interval]))
    deltaVy.append(HostGalaxies[n,7] - (assoc[:,8][interval]))
    deltaVz.append(HostGalaxies[n,8] - (assoc[:,9][interval]))
    count = count + i

# Creating the list of sublists containing 3 elements each: deltaVx, deltaVy and deltaVz.

deltaVall = []
for i in np.arange(len(HostGalaxies)):
    AllVelocities = np.vstack([deltaVx[i], deltaVy[i], deltaVz[i]])
    deltaVall.append(AllVelocities)
foo = np.hstack(deltaVall)
np.shape(foo)
FinalVall = foo.T
FirstInterval = np.concatenate((FinalVall[:,0], FinalVall[:,1], FinalVall[:,2]))
#print 'The First Interval Has', len(NewFinalFIRST), 'Satellites'

#---------------------------------------------------------------------------------------------------------------

# SECOND RADIUS INTERVAL

deltaVx = []
deltaVy = []
deltaVz = []

count = 0

# Calculating deltaVx, deltaVy and deltaVz

for n, i in enumerate(HostGalaxies[:,13].astype('int')):
        
    assoc = SatelliteGalaxies[count:count+i,:]
    
    interval = (assoc[:,6] >= intervals[1]) & (assoc[:,6] <= intervals[2])
        
    deltaVx.append(HostGalaxies[n,6] - (assoc[:,7][interval]))
    deltaVy.append(HostGalaxies[n,7] - (assoc[:,8][interval]))
    deltaVz.append(HostGalaxies[n,8] - (assoc[:,9][interval]))
    count = count + i

# Creating the list of sublists containing 3 elements each: deltaVx, deltaVy and deltaVz.
    
deltaVall = []
for i in np.arange(len(HostGalaxies)):
    AllVelocities = np.vstack([deltaVx[i], deltaVy[i], deltaVz[i]])
    deltaVall.append(AllVelocities)
foo = np.hstack(deltaVall)
np.shape(foo)
FinalVall = foo.T
SecondInterval = np.concatenate((FinalVall[:,0], FinalVall[:,1], FinalVall[:,2]))
#print 'The Second Interval Has', len(NewFinalSECOND), 'Satellites'

#---------------------------------------------------------------------------------------------------------------

# THIRD RADIUS INTERVAL

deltaVx = []
deltaVy = []
deltaVz = []

count = 0

# Calculating deltaVx, deltaVy and deltaVz

for n, i in enumerate(HostGalaxies[:,13].astype('int')):
        
    assoc = SatelliteGalaxies[count:count+i,:]
    
    interval = (assoc[:,6] >= intervals[2]) & (assoc[:,6] <= intervals[3])
        
    deltaVx.append(HostGalaxies[n,6] - (assoc[:,7][interval]))
    deltaVy.append(HostGalaxies[n,7] - (assoc[:,8][interval]))
    deltaVz.append(HostGalaxies[n,8] - (assoc[:,9][interval]))
    count = count + i

# Creating the list of sublists containing 3 elements each: deltaVx, deltaVy and deltaVz.
    
deltaVall = []
for i in np.arange(len(HostGalaxies)):
    AllVelocities = np.vstack([deltaVx[i], deltaVy[i], deltaVz[i]])
    deltaVall.append(AllVelocities)
foo = np.hstack(deltaVall)
np.shape(foo)
FinalVall = foo.T
ThirdInterval = np.concatenate((FinalVall[:,0], FinalVall[:,1], FinalVall[:,2]))
#print 'The Third Interval Has', len(NewFinalTHIRD), 'Satellites'

#-------------------------------------------------------------------------------------------------------------------

# FOURTH RADIUS INTERVAL

deltaVx = []
deltaVy = []
deltaVz = []

count = 0

# Calculating deltaVx, deltaVy and deltaVz

for n, i in enumerate(HostGalaxies[:,13].astype('int')):
        
    assoc = SatelliteGalaxies[count:count+i,:]
    
    interval = (assoc[:,6] >= intervals[3]) & (assoc[:,6] <= intervals[4])
        
    deltaVx.append(HostGalaxies[n,6] - (assoc[:,7][interval]))
    deltaVy.append(HostGalaxies[n,7] - (assoc[:,8][interval]))
    deltaVz.append(HostGalaxies[n,8] - (assoc[:,9][interval]))
    count = count + i

# Creating the list of sublists containing 3 elements each: deltaVx, deltaVy and deltaVz.
    
deltaVall = []
for i in np.arange(len(HostGalaxies)):
    AllVelocities = np.vstack([deltaVx[i], deltaVy[i], deltaVz[i]])
    deltaVall.append(AllVelocities)
foo = np.hstack(deltaVall)
np.shape(foo)
FinalVall = foo.T
FourthInterval = np.concatenate((FinalVall[:,0], FinalVall[:,1], FinalVall[:,2]))
#print 'The Fourth Interval Has', len(NewFinalFOURTH), 'Satellites'

#---------------------------------------------------------------------------------------------------------------

# FIFTH RADIUS INTERVAL

deltaVx = []
deltaVy = []
deltaVz = []

count = 0

# Calculating deltaVx, deltaVy and deltaVz

for n, i in enumerate(HostGalaxies[:,13].astype('int')):
        
    assoc = SatelliteGalaxies[count:count+i,:]
    
    interval = (assoc[:,6] >= intervals[4]) & (assoc[:,6] <= intervals[5])
        
    deltaVx.append(HostGalaxies[n,6] - (assoc[:,7][interval]))
    deltaVy.append(HostGalaxies[n,7] - (assoc[:,8][interval]))
    deltaVz.append(HostGalaxies[n,8] - (assoc[:,9][interval]))
    count = count + i

# Creating the list of sublists containing 3 elements each: deltaVx, deltaVy and deltaVz.
    
deltaVall = []
for i in np.arange(len(HostGalaxies)):
    AllVelocities = np.vstack([deltaVx[i], deltaVy[i], deltaVz[i]])
    deltaVall.append(AllVelocities)
foo = np.hstack(deltaVall)
np.shape(foo)
FinalVall = foo.T
FifthInterval = np.concatenate((FinalVall[:,0], FinalVall[:,1], FinalVall[:,2]))
#print 'The Fifth Interval Has', len(NewFinalFIFTH), 'Satellites'

#---------------------------------------------------------------------------------------------------------------------

# SIXTH RADIUS INTERVAL

deltaVx = []
deltaVy = []
deltaVz = []

count = 0

# Calculating deltaVx, deltaVy and deltaVz

for n, i in enumerate(HostGalaxies[:,13].astype('int')):
        
    assoc = SatelliteGalaxies[count:count+i,:]
    
    interval = (assoc[:,6] >= intervals[5]) & (assoc[:,6] <= intervals[6])
        
    deltaVx.append(HostGalaxies[n,6] - (assoc[:,7][interval]))
    deltaVy.append(HostGalaxies[n,7] - (assoc[:,8][interval]))
    deltaVz.append(HostGalaxies[n,8] - (assoc[:,9][interval]))
    count = count + i

# Creating the list of sublists containing 3 elements each: deltaVx, deltaVy and deltaVz.
    
deltaVall = []
for i in np.arange(len(HostGalaxies)):
    AllVelocities = np.vstack([deltaVx[i], deltaVy[i], deltaVz[i]])
    deltaVall.append(AllVelocities)
foo = np.hstack(deltaVall)
np.shape(foo)
FinalVall = foo.T
SixthInterval = np.concatenate((FinalVall[:,0], FinalVall[:,1], FinalVall[:,2]))
#print 'The Sixth Interval Has', len(NewFinalSIXTH), 'Satellites'

#--------------------------------------------------------------------------------------------------------------------

# SEVENTH RADIUS INTERVAL

deltaVx = []
deltaVy = []
deltaVz = []

count = 0

# Calculating deltaVx, deltaVy and deltaVz

for n, i in enumerate(HostGalaxies[:,13].astype('int')):
        
    assoc = SatelliteGalaxies[count:count+i,:]
    
    interval = (assoc[:,6] >= intervals[6]) & (assoc[:,6] <= intervals[7])
        
    deltaVx.append(HostGalaxies[n,6] - (assoc[:,7][interval]))
    deltaVy.append(HostGalaxies[n,7] - (assoc[:,8][interval]))
    deltaVz.append(HostGalaxies[n,8] - (assoc[:,9][interval]))
    count = count + i

# Creating the list of sublists containing 3 elements each: deltaVx, deltaVy and deltaVz.
    
deltaVall = []
for i in np.arange(len(HostGalaxies)):
    AllVelocities = np.vstack([deltaVx[i], deltaVy[i], deltaVz[i]])
    deltaVall.append(AllVelocities)
foo = np.hstack(deltaVall)
np.shape(foo)
FinalVall = foo.T
SeventhInterval = np.concatenate((FinalVall[:,0], FinalVall[:,1], FinalVall[:,2]))
#print 'The Seventh Interval Has', len(NewFinalSEVENTH), 'Satellites'

#----------------------------------------------------------------------------------------------------------------------


# Creating the Host-Satellite Velocity Difference Histogram for the First Interval (0-1)



#entries, edges = np.histogram(NewFinalFIRST, bins = 36)
#nentries , nedges = np.histogram(NewFinalFIRST, bins = 36, normed = True)

#yerr = np.sqrt(entries)/(np.sum(entries)*np.diff(edges)[0])

#bin_centers = 0.5*(edges[:-1] + edges[1:])

#plt.hist(FirstInterval, bins = 36, normed = True, histtype = 'step')
#plt.errorbar(bin_centers, nentries, yerr = yerr, elinewidth = 1, fmt = 'r.',markersize = 0.8, capsize = 2)
#plt.xlabel(r'$\Delta V$', fontsize = 9)
#plt.ylabel('Probability ['r'f($\Delta V)$''] of Finding a Satellite with a Given Value of 'r'$\Delta V$', fontsize = 7)
#plt.title('Histogram of the Number of Satellites with a Given Value of 'r'$\Delta V$' ' \n (3D Radius of Satellites in Units of Host Virial Radius)', fontsize = 9)
#plt.text(x = -1550, y = 0.0020,s='3D Radius of Satellites:', fontsize = 7.5)
#plt.text(x= -1550, y = 0.00190, s = '0.0014'r'$\leq$''3D Radius'r'$\leq$''0.3816', fontsize = 6.1)
#plt.text(x = 750, y = 0.00199, s = '5346 Satellites', fontsize = 7)
#plt.rcParams['xtick.labelsize']=6
#plt.rcParams['ytick.labelsize']=6
#plt.savefig('subplot0.eps')


# Creating the Plot for the Second Interval (1-2)

#entries, edges = np.histogram(NewFinalSECOND, bins = 36)
#nentries , nedges = np.histogram(NewFinalSECOND, bins = 36, normed = True)

#yerr = np.sqrt(entries)/(np.sum(entries)*np.diff(edges)[0])

#bin_centers = 0.5*(edges[:-1] + edges[1:])

#plt.hist(SecondInterval, bins = 36, normed = True, histtype = 'step')
#plt.errorbar(bin_centers, nentries, yerr = yerr, elinewidth = 1, fmt = 'r.',markersize = 0.8, capsize = 2)
#plt.xlabel(r'$\Delta V$', fontsize = 9)
#plt.ylabel('Probability ['r'f($\Delta V)$''] of Finding a Satellite with a Given Value of 'r'$\Delta V$', fontsize = 7)
#plt.text(x = -1300, y = 0.00175,s='3D Radius of Satellites:', fontsize = 7.5)
#plt.text(x= -1300, y = 0.00165, s = '0.2869'r'$\leq$''3D Radius'r'$\leq$''0.5724', fontsize = 6.1)
#plt.text(x = 750, y = 0.00174, s = '7143 Satellites', fontsize = 7)
#plt.rcParams['xtick.labelsize'] = 6
#plt.rcParams['ytick.labelsize'] = 6
#plt.savefig('subplot1.eps')
#x = stats.norm.fit(NewFinalSECOND, loc = 0)
#y = stats.norm.pdf(np.arange(-1500, 3000, 1), loc = 0, scale = 1)
#print(x)
    
# Creating the Plot for the Third Interval (2-3)

#entries, edges = np.histogram(NewFinalTHIRD, bins = 36)
#nentries , nedges = np.histogram(NewFinalTHIRD, bins = 36, normed = True)

#yerr = np.sqrt(entries)/(np.sum(entries)*np.diff(edges)[0])

#bin_centers = 0.5*(edges[:-1] + edges[1:])

#plt.hist(ThirdInterval, bins = 36, normed = True, histtype = 'step')
#plt.errorbar(bin_centers, nentries, yerr = yerr, elinewidth = 1, fmt = 'r.',markersize = 0.8, capsize = 2)
#plt.xlabel(r'$\Delta V$', fontsize = 9)
#plt.ylabel('Probability ['r'f($\Delta V)$''] of Finding a Satellite with a Given Value of 'r'$\Delta V$', fontsize = 7)
#plt.text(x = -1500, y = 0.00175,s='3D Radius of Satellites:', fontsize = 7.5)
#plt.text(x= -1500, y = 0.00165, s = '0.5724'r'$\leq$''3D Radius'r'$\leq$''0.8579', fontsize = 6.1)
#plt.text(x = 750, y = 0.00174, s = '8658 Satellites', fontsize = 7)
#plt.rcParams['xtick.labelsize'] = 6
#plt.rcParams['ytick.labelsize'] = 6
#plt.savefig('subplot2.eps')


# Creating the Plot for the 4th Interval (3-4)

#entries, edges = np.histogram(NewFinalFOURTH, bins = 36)
#nentries , nedges = np.histogram(NewFinalFOURTH, bins = 36, normed = True)

#yerr = np.sqrt(entries)/(np.sum(entries)*np.diff(edges)[0])

#bin_centers = 0.5*(edges[:-1] + edges[1:])

#plt.hist(FourthInterval, bins = 36, normed = True, histtype = 'step')
#plt.errorbar(bin_centers, nentries, yerr = yerr, elinewidth = 1, fmt = 'r.',markersize = 0.8, capsize = 2)
#plt.xlabel(r'$\Delta V$', fontsize = 9)
#plt.ylabel('Probability ['r'f($\Delta V)$''] of Finding a Satellite with a Given Value of 'r'$\Delta V$', fontsize = 7)
#plt.text(x = -1500, y = 0.0020, s='3D Radius of Satellites:', fontsize = 7.5)
#plt.text(x= -1500, y = 0.00190, s = '0.8579'r'$\leq$''3D Radius'r'$\leq$''1.1434', fontsize = 6.1)
#plt.text(x = 750, y = 0.00199 , s = '8187 Satellites', fontsize = 7)
#plt.rcParams['xtick.labelsize'] = 6
#plt.rcParams['ytick.labelsize'] = 6
#plt.savefig('subplot3.eps')

# Creating the Plot for the 5th Interval (4-5)

#entries, edges = np.histogram(NewFinalFIFTH, bins = 36)
#nentries , nedges = np.histogram(NewFinalFIFTH, bins = 36, normed = True)

#yerr = np.sqrt(entries)/(np.sum(entries)*np.diff(edges)[0])

#bin_centers = 0.5*(edges[:-1] + edges[1:])

#plt.hist(FifthInterval, bins = 36, normed = True, histtype = 'step')
#plt.errorbar(bin_centers, nentries, yerr = yerr, elinewidth = 1, fmt = 'r.',markersize = 0.8, capsize = 2)
#plt.xlabel(r'$\Delta V$', fontsize = 9)
#plt.ylabel('Probability ['r'f($\Delta V)$''] of Finding a Satellite with a Given Value of 'r'$\Delta V$', fontsize = 7)
#plt.text(x = -1500, y = 0.0020, s='3D Radius of Satellites:', fontsize = 7.5)
#plt.text(x= -1500, y = 0.00190, s = '1.1434'r'$\leq$''3D Radius'r'$\leq$''1.4289', fontsize = 6.1)
#plt.text(x = 750, y = 0.00199, s = '7788 Satellites', fontsize = 7)
#plt.rcParams['xtick.labelsize'] = 6
#plt.rcParams['ytick.labelsize'] = 6
#plt.savefig('subplot4.eps')


# Creating the Plot for the 6th Interval (5-6)

#entries, edges = np.histogram(NewFinalSIXTH, bins = 36)
#nentries , nedges = np.histogram(NewFinalSIXTH, bins = 36, normed = True)

#yerr = np.sqrt(entries)/(np.sum(entries)*np.diff(edges)[0])

#bin_centers = 0.5*(edges[:-1] + edges[1:])

#plt.hist(SixthInterval, bins = 36, normed = True, histtype = 'step')
#plt.errorbar(bin_centers, nentries, yerr = yerr, elinewidth = 1, fmt = 'r.',markersize = 0.8, capsize = 2)
#plt.xlabel(r'$\Delta V$', fontsize = 9)
#plt.ylabel('Probability ['r'f($\Delta V)$''] of Finding a Satellite with a Given Value of 'r'$\Delta V$', fontsize = 7)
#plt.text(x = -1200, y = 0.0020, s='3D Radius of Satellites:', fontsize = 7.5)
#plt.text(x= -1200, y = 0.00190, s = '1.4289'r'$\leq$''3D Radius'r'$\leq$''1.7144', fontsize = 6.1)
#plt.text(x = 750, y = 0.00199, s = '7107 Satellites', fontsize = 7)
#plt.rcParams['xtick.labelsize'] = 6
#plt.rcParams['ytick.labelsize'] = 6
#plt.savefig('subplot5.eps')

    
# Creating the Plot for the 7th Interval (6-7)

#entries, edges = np.histogram(NewFinalSEVENTH, bins = 36)
#nentries , nedges = np.histogram(NewFinalSEVENTH, bins = 36, normed = True)

#yerr = np.sqrt(entries)/(np.sum(entries)*np.diff(edges)[0])

#bin_centers = 0.5*(edges[:-1] + edges[1:])

#plt.hist(SeventhInterval, bins = 36, normed = True, histtype = 'step')
#plt.errorbar(bin_centers, nentries, yerr = yerr, elinewidth = 1, fmt = 'r.',markersize = 0.8, capsize = 2)
#plt.xlabel(r'$\Delta V$', fontsize = 9)
#plt.ylabel('Probability ['r'f($\Delta V)$''] of Finding a Satellite with a Given Value of 'r'$\Delta V$', fontsize = 7)
#plt.text(x = -1200, y = 0.0020, s='3D Radius of Satellites:', fontsize = 7.5)
#plt.text(x= -1200, y = 0.00190, s = '1.7144'r'$\leq$''3D Radius'r'$\leq$''1.9999', fontsize = 6.1)
#plt.text(x = 650, y = 0.00199, s = '6318 Satellites', fontsize = 7)
#plt.rcParams['xtick.labelsize'] = 6
#plt.rcParams['ytick.labelsize'] = 6
#plt.savefig('subplot6.eps')



fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#fig1.suptitle('Histogram of the Probability Distribution of Satellites with a Given Value of 'r'$\Delta V$' ' \n (3D Radius of Satellites in Units of Host Virial Radius - 7221 Satellites in Each Plot)', fontsize = 8)
entries_0, edges_0 = np.histogram(NewFinalFIRST, bins = 10000)
nentries_0 , nedges_0 = np.histogram(NewFinalFIRST, bins = 10000, normed = True)
Aentries_0, Aedges_0 = np.histogram(NewFinalFIRST, bins = 36)
Mentries_0, Medges_0 = np.histogram(NewFinalFIRST, bins = 36, normed = True)

entries_1, edges_1 = np.histogram(NewFinalSECOND, bins = 10000)
nentries_1 , nedges_1 = np.histogram(NewFinalSECOND, bins = 10000, normed = True)
Aentries_1, Aedges_1 = np.histogram(NewFinalSECOND, bins = 36)
Mentries_1, Medges_1 = np.histogram(NewFinalSECOND, bins = 36, normed = True)

entries_2, edges_2 = np.histogram(NewFinalTHIRD, bins = 10000)
nentries_2 , nedges_2 = np.histogram(NewFinalTHIRD, bins = 10000, normed = True)
Aentries_2, Aedges_2 = np.histogram(NewFinalTHIRD, bins = 36)
Mentries_2, Medges_2 = np.histogram(NewFinalTHIRD, bins = 36, normed = True)

entries_3, edges_3 = np.histogram(NewFinalFOURTH, bins = 10000)
nentries_3 , nedges_3 = np.histogram(NewFinalFOURTH, bins = 10000, normed = True)
Aentries_3, Aedges_3 = np.histogram(NewFinalFOURTH, bins = 36)
Mnentries_3 , Mnedges_3 = np.histogram(NewFinalFOURTH, bins = 36, normed = True)

yerr_0 = np.sqrt(entries_0)/(np.sum(entries_0)*np.diff(edges_0)[0])
yerr_1 = np.sqrt(entries_1)/(np.sum(entries_1)*np.diff(edges_1)[0])
yerr_2 = np.sqrt(entries_2)/(np.sum(entries_2)*np.diff(edges_2)[0])
yerr_3 = np.sqrt(entries_3)/(np.sum(entries_3)*np.diff(edges_3)[0])

Ayerr_0 = np.sqrt(Aentries_0)/(np.sum(Aentries_0)*np.diff(Aedges_0)[0])
Ayerr_1 = np.sqrt(Aentries_1)/(np.sum(Aentries_1)*np.diff(Aedges_1)[0])
Ayerr_2 = np.sqrt(Aentries_2)/(np.sum(Aentries_2)*np.diff(Aedges_2)[0])
Ayerr_3 = np.sqrt(Aentries_3)/(np.sum(Aentries_3)*np.diff(Aedges_3)[0])

bin_centers_0 = 0.5*(edges_0[:-1] + edges_0[1:])
bin_centers_1 = 0.5*(edges_1[:-1] + edges_1[1:])
bin_centers_2 = 0.5*(edges_2[:-1] + edges_2[1:])
bin_centers_3 = 0.5*(edges_3[:-1] + edges_3[1:])

Mbin_centers_0 = 0.5*(Aedges_0[:-1] + Aedges_0[1:])
Mbin_centers_1 = 0.5*(Aedges_1[:-1] + Aedges_1[1:])
Mbin_centers_2 = 0.5*(Aedges_2[:-1] + Aedges_2[1:])
Mbin_centers_3 = 0.5*(Aedges_3[:-1] + Aedges_3[1:])

# Axis 1

ax1.hist(NewFinalFIRST, bins = 36, normed = True, histtype = 'step', linewidth = 0.2)
g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, bin_centers_0, nentries_0)
ax1.plot(bin_centers_0, g(bin_centers_0), label = 'Gaussian', linewidth = 0.2)

l_init = models.Lorentz1D(amplitude=0.75, x_0 =0.001, fwhm = 1000)
fit_l = fitting.LevMarLSQFitter()
l = fit_l(l_init, bin_centers_0, nentries_0)
ax1.plot(bin_centers_0, l(bin_centers_0), label = 'Lorentzian', linewidth = 0.2)
#ax1.errorbar(Mbin_centers_0, Mentries_0, yerr = Ayerr_0, elinewidth = 0.2, fmt = 'r.',markersize = 0.1, capsize = 1)
ax1.text(x = -1650, y = 0.0025,s='3D Host-Satellite Distance:', fontsize = 3.7)
ax1.text(x= -1680, y = 0.00235, s = '0.0014'r'$\leq$''Distance [kpc]'r'$\leq$''0.3816', fontsize = 3.2)
ax1.text(x= 1150, y = 0.00229, s = 'Plot #1', fontsize = 5.5)
#ax1.text(x = 900, y = 0.00079, s = '7221 Satellites', fontsize = 3.7)
ax1.set_ylim(0,0.0030)
ax1.set_xlim(-2000, 2000)
ax1.tick_params(axis = 'both', labelsize = 4.5)
ax1.set_ylabel(r'P($\Delta V)$', fontsize = 9)
ax1.legend(loc = 0, prop = {'size': 4.5})
#fig1.set_figheight(9)
#fig1.set_figwidth(9)

# Axis 2

ax2.hist(NewFinalSECOND, bins = 36, normed = True, histtype = 'step', linewidth = 0.2)
g_init1 = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g1 = fitting.LevMarLSQFitter()
g1 = fit_g1(g_init1, bin_centers_1, nentries_1)
ax2.plot(bin_centers_1, g1(bin_centers_1), label = 'Gaussian', linewidth = 0.2)

l_init1 = models.Lorentz1D(amplitude=0.75, x_0 = 0, fwhm = 1000)
fit_l1 = fitting.LevMarLSQFitter()
l1 = fit_l1(l_init1, bin_centers_1, nentries_1)
ax2.plot(bin_centers_1, l1(bin_centers_1), label = 'Lorentzian', linewidth = 0.2)
#ax2.errorbar(Mbin_centers_1, Mentries_1, yerr = Ayerr_1, elinewidth = 0.2, fmt = 'r.',markersize = 0.1, capsize = 1)
ax2.text(x = -1650, y = 0.0025,s='3D Host-Satellite Distance:', fontsize = 3.7)
ax2.text(x= -1680, y = 0.00235, s = '0.3816'r'$\leq$''Distance [kpc]'r'$\leq$''0.6406', fontsize = 3.2)
ax2.text(x= 1150, y = 0.00229, s = 'Plot #2', fontsize = 5.5)
ax2.set_xlim(-2000,2000)
ax2.set_ylim(0,0.0030)
ax2.legend(loc = 0, prop = {'size': 4.5})
ax2.tick_params(axis = 'both', labelsize = 4.5)

# Axis 3 

ax3.hist(NewFinalTHIRD, bins = 36, normed = True, histtype = 'step', linewidth = 0.2)
g_init2 = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g2 = fitting.LevMarLSQFitter()
g2 = fit_g2(g_init2, bin_centers_2, nentries_2)
ax3.plot(bin_centers_2, g2(bin_centers_2), label = 'Gaussian', linewidth = 0.2)

l_init2 = models.Lorentz1D(amplitude=0.75, x_0 = 0, fwhm = 1000)
fit_l2 = fitting.LevMarLSQFitter()
l2 = fit_l2(l_init2, bin_centers_2, nentries_2)
ax3.plot(bin_centers_2, l2(bin_centers_2), label = 'Lorentzian', linewidth = 0.2)
#ax3.errorbar(bin_centers_2, nentries_2, yerr = yerr_2, elinewidth = 0.2, fmt = 'r.',markersize = 0.1, capsize = 1)
ax3.text(x = -1650, y = 0.0025,s='3D Host-Satellite Distance:', fontsize = 3.7)
ax3.text(x= -1680, y = 0.00235, s = '0.6406'r'$\leq$''Distance [kpc]'r'$\leq$''0.8731', fontsize = 3.2)
ax3.text(x= 1150, y = 0.00229, s = 'Plot #3', fontsize = 5.5)
ax3.set_ylim(0,0.0030)
ax3.set_xlim(-2000, 2000)
ax3.tick_params(axis = 'both', labelsize = 4.5)
ax3.set_xlabel(r'$\Delta V$', fontsize = 9)
ax3.legend(loc = 0, prop = {'size': 4.5})
ax3.set_ylabel(r'P($\Delta V)$', fontsize = 9)

# Axis 4

ax4.hist(NewFinalFOURTH, bins = 36, normed = True, histtype = 'step', linewidth = 0.2)
g_init3 = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g3 = fitting.LevMarLSQFitter()
g3 = fit_g3(g_init3, bin_centers_3, nentries_3)
ax4.plot(bin_centers_3, g3(bin_centers_3), label = 'Gaussian', linewidth = 0.2)

l_init3 = models.Lorentz1D(amplitude=0.75, x_0 = 0, fwhm = 1000)
fit_l3 = fitting.LevMarLSQFitter()
l3 = fit_l3(l_init3, bin_centers_3, nentries_3)
ax4.plot(bin_centers_3, l3(bin_centers_3), label = 'Lorentzian', linewidth = 0.2)


#ax4.errorbar(bin_centers_3, nentries_3, yerr = yerr_3, elinewidth = 0.2, fmt = 'r.',markersize = 0.1, capsize = 1)
ax4.text(x = -1650, y = 0.0025, s='3D Host-Satellite Distance:', fontsize = 3.7)
ax4.text(x= -1680, y = 0.00235, s = '0.8731'r'$\leq$''Distance [kpc]'r'$\leq$''1.1250', fontsize = 3.2)
ax4.text(x= 1150, y = 0.00229, s = 'Plot #4', fontsize = 5.5)
ax4.set_ylim(0,0.0030)
ax4.set_xlim(-2000, 2000)
ax4.tick_params(axis = 'both', labelsize = 4.5)
ax4.legend(loc = 0, prop = {'size': 4.5})
ax4.set_xlabel(r'$\Delta V$', fontsize = 9)



fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
#fig2.suptitle('Histogram of the Probability Distribution of Satellites with a Given Value of 'r'$\Delta V$' ' \n (3D Radius of Satellites in Units of Host Virial Radius - 7221 Satellites in Each Plot)', fontsize = 8)

entries_4, edges_4 = np.histogram(NewFinalFIFTH, bins = 10000)
nentries_4 , nedges_4 = np.histogram(NewFinalFIFTH, bins = 10000, normed = True)
Aentries_4, Aedges_4 = np.histogram(NewFinalFIFTH, bins = 36)
Mentries_4, Medges_4 = np.histogram(NewFinalFIFTH, bins = 36, normed = True)


entries_5, edges_5 = np.histogram(NewFinalSIXTH, bins = 10000)
nentries_5 , nedges_5 = np.histogram(NewFinalSIXTH, bins = 10000, normed = True)
Aentries_5, Aedges_5 = np.histogram(NewFinalSIXTH, bins = 36)
Mentries_5, Medges_5 = np.histogram(NewFinalSIXTH, bins = 36, normed = True)

entries_6, edges_6 = np.histogram(NewFinalSEVENTH, bins = 10000)
nentries_6 , nedges_6 = np.histogram(NewFinalSEVENTH, bins = 10000, normed = True)
Aentries_6, Aedges_6 = np.histogram(NewFinalSEVENTH, bins = 36)
Mentries_6, Medges_6 = np.histogram(NewFinalSEVENTH, bins = 36, normed = True)

yerr_4 = np.sqrt(entries_4)/(np.sum(entries_4)*np.diff(edges_4)[0])
yerr_5 = np.sqrt(entries_5)/(np.sum(entries_5)*np.diff(edges_5)[0])
yerr_6 = np.sqrt(entries_6)/(np.sum(entries_6)*np.diff(edges_6)[0])

Ayerr_4 = np.sqrt(Aentries_4)/(np.sum(Aentries_4)*np.diff(Aedges_4)[0])
Ayerr_5 = np.sqrt(Aentries_5)/(np.sum(Aentries_5)*np.diff(Aedges_5)[0])
Ayerr_6 = np.sqrt(Aentries_6)/(np.sum(Aentries_6)*np.diff(Aedges_6)[0])


bin_centers_4 = 0.5*(edges_4[:-1] + edges_4[1:])
bin_centers_5 = 0.5*(edges_5[:-1] + edges_5[1:])
bin_centers_6 = 0.5*(edges_6[:-1] + edges_6[1:])

Mbin_centers_4 = 0.5*(Aedges_4[:-1] + Aedges_4[1:])
Mbin_centers_5 = 0.5*(Aedges_5[:-1] + Aedges_5[1:])
Mbin_centers_6 = 0.5*(Aedges_6[:-1] + Aedges_6[1:])

# Axis 5

ax5.hist(NewFinalFIFTH, bins = 36, normed = True, histtype = 'step', linewidth = 0.2)

g_init4 = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g4 = fitting.LevMarLSQFitter()
g4 = fit_g4(g_init4, bin_centers_4, nentries_4)
ax5.plot(bin_centers_4, g4(bin_centers_4), label = 'Gaussian', linewidth = 0.2)

l_init4 = models.Lorentz1D(amplitude=0.75, x_0 = 0, fwhm = 1000)
fit_l4 = fitting.LevMarLSQFitter()
l4 = fit_l4(l_init4, bin_centers_4, nentries_4)
ax5.plot(bin_centers_4, l4(bin_centers_4), label = 'Lorentzian', linewidth = 0.2)

#ax5.errorbar(bin_centers_4, nentries_4, yerr = yerr_4, elinewidth = 0.2, fmt = 'r.',markersize = 0.1, capsize = 1)
ax5.text(x = -1650, y = 0.0030, s='3D Host-Satellite Distance:', fontsize = 3.7)
ax5.text(x= -1680, y = 0.0028, s = '1.1250'r'$\leq$''Distance [kpc]'r'$\leq$''1.3905', fontsize = 3.2)
ax5.text(x= 1150, y = 0.00385, s = 'Plot #5', fontsize = 5.5)
ax5.set_ylim(0,0.0050)
ax5.set_xlim(-2000, 2000)
ax5.tick_params(axis = 'both', labelsize = 4.5)
ax5.set_ylabel(r'P($\Delta V)$', fontsize = 9)
ax5.legend(loc = 0, prop = {'size': 4.5})

ax6.hist(NewFinalSIXTH, bins = 36, normed = True, histtype = 'step', linewidth = 0.2)

g_init5 = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g5 = fitting.LevMarLSQFitter()
g5 = fit_g5(g_init5, bin_centers_5, nentries_5)
ax6.plot(bin_centers_5, g5(bin_centers_5), label = 'Gaussian', linewidth = 0.2)

l_init5 = models.Lorentz1D(amplitude=0.75, x_0 = 0, fwhm = 1000)
fit_l5 = fitting.LevMarLSQFitter()
l5 = fit_l5(l_init5, bin_centers_5, nentries_5)
ax6.plot(bin_centers_5, l5(bin_centers_5), label = 'Lorentzian', linewidth = 0.2)

#ax6.errorbar(bin_centers_5, nentries_5, yerr = yerr_5, elinewidth = 0.2, fmt = 'r.',markersize = 0.1, capsize = 1)
ax6.text(x = -1650, y = 0.0030, s='3D Host-Satellite Distance:', fontsize = 3.7)
ax6.text(x= -1680, y = 0.0028, s = '1.3905'r'$\leq$''Distance [kpc]'r'$\leq$''1.6755', fontsize = 3.2)
ax6.text(x= 1150, y = 0.00385, s = 'Plot #6', fontsize = 5.5)
ax6.set_ylim(0,0.0050)
ax6.set_xlim(-2000, 2000)

ax6.tick_params(axis = 'both', labelsize = 4.5)
ax6.legend(loc = 0, prop = {'size': 4.5})


ax7.hist(NewFinalSEVENTH, bins = 36, normed = True, histtype = 'step', linewidth = 0.2)

g_init6 = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
fit_g6 = fitting.LevMarLSQFitter()
g6 = fit_g6(g_init6, bin_centers_6, nentries_6)
ax7.plot(bin_centers_6, g6(bin_centers_6), label = 'Gaussian', linewidth = 0.2)

l_init6 = models.Lorentz1D(amplitude=0.75, x_0 = 0, fwhm = 1000)
fit_l6 = fitting.LevMarLSQFitter()
l6 = fit_l6(l_init6, bin_centers_6, nentries_6)
ax7.plot(bin_centers_6, l5(bin_centers_6), label = 'Lorentzian', linewidth = 0.2)

#ax7.errorbar(bin_centers_6, nentries_6, yerr = yerr_6, elinewidth = 0.2, fmt = 'r.',markersize = 0.1, capsize = 1)
ax7.text(x = -1650, y = 0.0030, s='3D Host-Satellite Distance:', fontsize = 3.7)
ax7.text(x = -1680, y = 0.0028, s = '1.6755'r'$\leq$''Distance [kpc]'r'$\leq$''1.9990', fontsize = 3.2)
ax7.text(x= 1150, y = 0.00385, s = 'Plot #7', fontsize = 5.5)
ax7.set_ylim(0,0.0050)
ax7.set_xlim(-2000, 2000)
ax7.tick_params(axis = 'both', labelsize = 4.5)
ax7.set_xlabel(r'$\Delta V$'' [km/s]', fontsize = 9)
ax7.set_ylabel(r'P($\Delta V)$', fontsize = 9)
ax7.legend(loc = 0, prop = {'size': 4.5})

#ax8.text(x = 0.25, y = 0.5, s = 'Ignore This Plot')
#ax8.set_xlabel(r'$\Delta V$', fontsize = 9)
ax8.tick_params(axis = 'both', labelsize = 4.5)
#ax8.set_xlabel(r'$\Delta V$', fontsize = 9)


fig1.savefig('First4.pdf')
fig2.savefig('Second3.pdf')
#DATA = column_stack([bin_centers, entries, np.sqrt(entries)])

#comments4 = ['dline 1', 'dline 2']
#labels4 = ['Bin Centres of Delta V (for 3D Radius from 0.0014 to 0.2869)', 'Number of Satellites', 'Uncertainty/Error']
#label_row4 = '              '.join(labels4)
#myheader4 = '\n'.join(comments4)
#myheader4 += '\n'
#myheader4 += label_row4
#np.savetxt('HistogramTextFile.txt', DATA, delimiter = '                  ', header = myheader4)




