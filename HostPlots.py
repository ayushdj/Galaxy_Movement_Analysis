

#======================================================================================
""" Importing the necessary Python libraries 
"""

import numpy as np
import matplotlib.pyplot as plt

#======================================================================================
""" Reading the file, looping over the lines and defining 
    the required arrays 
"""

# Reading in the file 

f = open('HostsSats.txt', 'r')
lines = f.read().split(',')
f.close()
line = lines[0].split('\n')

# Setting the parameters to read in only the hosts

j = 0
HostGalaxies = [] # creating an empty array to put each host line of information
                  # into

# Looping over all 2174 Hosts

for i in range(2174):
    split = line[j].split(' ') # breaking the string up by 1 space each
    good = [len(x) > 0 for x in split] 
    split = np.array(split) # putting split into a numpy array
    HostGalaxies.append(split[good])
    
    # The last element of each host galaxy line is the number of satellites for 
    # that given host. So we need to make sure the loop skips over all the 
    # satellites for that given host in order to move on to the NEXT host. So 
    # if the first host galaxy has 651 lines (and therefore 651 satellites) after it, 
    # the loop needs to skip those 651 satellites and move onto the 652nd line
    # which is the next host, hence j needs to be updated with n + 1 where n 
    # is the number of satellites per host (i.e. the last number/piece of information 
    # in the host line)
        
    num = line[j][-3:]
    j += (int(num)+1)  

# turning HostGalaxies into a numpy array 
HostGalaxies = np.array(HostGalaxies[:]).astype('float')

# Calling the specific columns of data for the required insights

VelDisp = HostGalaxies[:,9] # The host galaxy's velocity dispersion
MaxCircularVel = HostGalaxies[:,10] # The maximum circular velocity of the host
HostRmag = HostGalaxies[:,5] # The host galaxy's R-band magnitude

#======================================================================================
""" Creating the necessary plots to get further insights into the movement and
    nature of the host galaxies 
"""

# Creating the plot of Number of Hosts as a function of the Velocity Dispersion  


plt.hist(VelDisp, bins = 50, histtype = 'step')
plt.xlabel('Velocity Dispersion (km/s)', fontsize = 9)
plt.ylabel('Number of Host Galaxies', fontsize = 9)
plt.title('Total Number of Host Galaxies as a Function of Velocity Dispersion (km/s)', fontsize = 9)
plt.tick_params(axis='x',which='minor',bottom=True)
plt.savefig('HostVelDisp.pdf')

# Creating the plot of the Stellar Mass against Velocity Dispersion

#plt.scatter(VelDisp, StellarMass, marker = 'o', s = 0.2)
#plt.xlabel('Velocity Dispersion (km/s)', fontsize = 9)
#plt.ylabel('Stellar Mass' ' ' r'$(M_S$'r'$_u$'r'$_n)$', fontsize = 9)
#plt.title('Host Galaxy Stellar Mass as a Function of Velocity Dispersion (Using Logarithmic Spacing)', fontsize = 9)
#plt.rcParams['xtick.labelsize']=7.5
#plt.rcParams['ytick.labelsize']=7.5
#plt.savefig('StellarMass.pdf')

# Creating the plot of Rmag against Velocity Dispersion

#plt.scatter(VelDisp, HostRmag, marker = 'o', s = 0.2)
#plt.gca().invert_yaxis()
#plt.xlabel('Velocity Dispersion (km/s)', fontsize = 9)
#plt.ylabel('r-band Absolute Magnitude ' ' 'r'$(M_r$' r'$)$''', fontsize = 9)
#plt.title('Host Absolute r-band Magnitude as a function of Velocity Dispersion', fontsize = 9)
#plt.savefig('HostRmagVeldisp.pdf')
