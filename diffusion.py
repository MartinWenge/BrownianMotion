#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time

print("diffusion of free particles")

#setup parameters
mu=0.0
sigma=1.0
dimension=2
diffusionConst=0.05
numSteps=40000#000
timestep=1
numParticles=1000 #00

#setup diffusion
diffusionCoeffi=np.sqrt(2*diffusionConst)

#loop over all single particles
for n in range(numParticles):
  #calculate path of one particle
  #setup coordinates = initial position
  coords=np.zeros(dimension)
  #setup path array, initial position at 0,0
  path=np.zeros(dimension)
  
  print("simulation of particle number "+str(n))
  for t in range(0,(numSteps-1),timestep):
    
    for idx, x_old in enumerate(coords):
      #update coordinates
      coords[idx]=(x_old+(diffusionCoeffi*np.random.normal(mu, sigma)))
    
    #save current position
    path=np.vstack((path,coords))  
    
  #calculate msd
  msd=np.zeros(path.shape[0])
  for idx, x in enumerate(path):
    #calculate current euclidic norm (x(t)-x(0))^2+(y(t)-y(0))^2
    currentmsd=0.0
    for i in range(dimension):
      currentmsd+=(x[i]-path[0][i])*(x[i]-path[0][i])
    
    # add this to the averaged msd container
    msd[idx]+=currentmsd

# normalize
np.divide(msd,numParticles)

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.loglog(range(msd.shape[0]), msd, 'ro')
ax1.loglog(range(msd.shape[0]), np.multiply(2*diffusionConst,range(msd.shape[0])), 'g-')
ax2.plot(range(msd.shape[0]), msd, 'ro')
ax2.plot(range(msd.shape[0]), np.multiply(2*diffusionConst,range(msd.shape[0])), 'g-')

plt.savefig("freeDiffusion.png")

#plt.show()
#time.sleep(5.0)
#print(msd)
