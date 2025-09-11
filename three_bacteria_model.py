# =============================================================================
# This is the code for modelling three bacteria
# Based on the single bacteria model
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import display, HTML

mu, sigma = 0, 0.5 # mean and standard deviation
numpoint = 1000
seed = 144
num_bacteria = 3
beads_per_bacterium = 3
num_beads = num_bacteria * beads_per_bacterium
rng = np.random.default_rng(seed) # random number generator
# Use normal distribution in number generator
random_1 = rng.normal(mu, sigma, size = (numpoint,num_beads,3))
random_2 = rng.normal(mu, sigma, size = (numpoint,num_beads,3))
random_3 = rng.normal(mu, sigma, size = (numpoint,num_beads,3))

radius = 0.5e-6 # Radius of the three circle represent bacteria
length = 4e-6
l_spring = (length - 2*radius)/2 # Calculate how long is the spring
spring_stiffness = 1e-5 # N/m, spring constant

def step1(v,h,a):
    """
    Input value: v- velocity at t - (1/2)h
            h - the time step
            a - the acceleration
    Output value: v = v + a * h
    """
    return v + a * h

def step2(v,f,random):
    """
    Input value: v - velocity
            f - a fraction, using this to reduce the velocity
            random - a random number from normal distribution
    Output value: delta v = -f * v_x + sqrt(f*(2-f)*(k_B*T/mass)) * random
    """
    return -f * v + np.sqrt(f * (2-f) * (k_B * T/mass)) * random

def step3(x,v,delta_v,h):
    """
    Input value: x - position at time t
            v - velocity
            delta_v - the value which we calculated in step 2
            h - time step
    Output value: x at time t + (1/2)h, x = x + (v + delta_v/2) * h
    """
    return x + (v + delta_v/2) * h

def step4(v,delta_v):
    """
    Input value: v_x - velocity
            delta_v - the value which we calculated in step 2
    Output value: v_x at time t + (1/2)h, v = v + delta_v
    """
    return v + delta_v

# Define a function on the force exerted by the spring

def spring_force(ri,rj,rk):
    """
    Input value: ri - the position of the middle beam
            rj - the position of first beam
            rz - the position of the last beam
    Output value: Force - force exert on the three beams in 3*3 matrix
    """
    x1 = ri - rj # Displacement of one spring
    x2 = rk - ri
    Force = np.zeros([3,3])
    F_ij = - (np.linalg.norm(x1) - l_spring) * spring_stiffness * (x1/np.linalg.norm(x1))
    F_ji = (np.linalg.norm(x1) - l_spring) * spring_stiffness * (x1/np.linalg.norm(x1))
    F_ki = - (np.linalg.norm(x2) - l_spring) * spring_stiffness * (x2/np.linalg.norm(x2))
    F_ik = (np.linalg.norm(x2) - l_spring) * spring_stiffness * (x2/np.linalg.norm(x2))
    Force[0] = F_ij + F_ik
    Force[1] = F_ji
    Force[2] = F_ki
    return Force
    
# Define a function on handling angle stiffness, and return the bending force
def bending_force(bond_stiffness,ri,rj,rk):
    ## i is the middle bead
    rij = rj-ri
    rik = rk-ri
    rij_abs = np.linalg.norm(rij)
    rik_abs = np.linalg.norm(rik)
    rijrik = rij_abs*rik_abs
    rij2 = rij_abs*rij_abs
    rik2 = rik_abs*rik_abs
    costhetajik = np.dot(rij,rik)/rijrik
    Force = np.zeros([3,3])
    i=1
    Force[i-1] = bond_stiffness*((rik+rij)/rijrik-costhetajik*(rij/rij2+rik/rik2))
    Force[i+1] = bond_stiffness*(costhetajik*rij/rij2-rik/rijrik)
    Force[i] = bond_stiffness*(costhetajik*rik/rik2-rij/rijrik)
    return Force

# Define Van der Waals force calculation
def van_der_waals_force(A_Ham,r_ab):
    """
    
    Parameters
    ----------
    r_ab : distance vector betwen a and b, unit: metre.
    A_Ham : Hamaker constant, unit: Joule.

    Returns
    -------
    Force : The Van der Waals force exert from a to b.

    """
    r_ab_scalar = np.linalg.norm(r_ab)
    Force = -(A_Ham/12) * radius/(r_ab_scalar**2 - 4 * radius * r_ab_scalar
                                 + 4 * radius**2) * (r_ab/r_ab_scalar)
    return Force

# Define initial values and arrays to hold values
bond_stiff = 1e-17 # Bond stiffness
mass = 1e-15
viscosity = 1e-3 # viscosity of water in 20 Celseius, Pa s
drag_coefficient = 6 * np.pi * radius * viscosity # Dragg coefficient
dt = 1e-5 # Time step
f = 1 - np.exp(-drag_coefficient * dt/mass)
k_B = 1.38e-23 # Boltzmann constant
T = 293.15 # Temperature, unit: K
A_Ham = 12.6e-21 # Hamaker constant for proteins, unit: Joule

# Array to hold position value for three bacteria with three beads
r_arr = np.zeros((numpoint, num_beads, 3))
#r_arr_2 = np.zeros((numpoint, num_beads, 3))
#r_arr_3 = np.zeros((numpoint, num_beads, 3))
# Array to hold velocity value for three bacteria with three beads
v_arr = np.zeros((numpoint, num_beads, 3))
#v_arr_2 = np.zeros((numpoint, num_beads, 3))
#v_arr_3 = np.zeros((numpoint, num_beads, 3))

ri = np.array([1.5,0,0]) * 1e-6
rj = np.array([0,0,0]) * 1e-6
rk = np.array([3,0,0]) * 1e-6

r_arr[0,0:3] = np.array([ri,rj,rk])
r_arr[0,3:6] = np.array([[1.5e-6,2e-6,0],
                         [0,2e-6,0],
                         [3e-6,2e-6,0]])
r_arr[0,6:9] = np.array([[1.5e-6,-2e-6,0],
                         [0,-2e-6,0],
                         [3e-6,-2e-6,0]])


i = 0

while i < numpoint - 1:
    # Calculate the force and acceleration
    # For Van der Waals force:
    # Define empty array for storing Van der Waals force
    F_van_arr = np.zeros((num_beads,3))
    #F_van_arr_2 = np.empty((3,num_beads))
    #F_van_arr_3 = np.empty((3,num_beads))
    for j in range(num_beads):
        ##print(f"The loop number is: {j} and {i}")
        heading = "***"
        for k in range(num_beads):
            # Looking at the other beads
            if j//beads_per_bacterium == k//beads_per_bacterium:
                # bead is in same bacterium, force = 0
                F_van = np.array([0.0,0.0,0.0])
                r_jk = np.array([0.0,0.0,0.0])
            else:
                # bead on other bacterium, compute force
        #if j <= 2:
            # The distance vector between the two beads from other two bacteria
                r_jk = r_arr[i,j,] - r_arr[i,k,]
            #r_31 = r_arr_1[i,j,] - r_arr_2[i,:]
            
            # The force experienced due to Van der Waals force from each beads 
            #for k in range(3):
                F_van = -van_der_waals_force(A_Ham, r_jk) #+ van_der_waals_force(A_Ham, r_31[k])
            F_van_arr[j] += F_van
            ##print(f"Distance r_jk is: {r_jk}")
            ##print(f"Force F_jk is: {F_van}")
            #print(r_arr_2[i,:])
        ##print(f"Force on bead {j} is {F_van_arr[j]}")
        """
        # If the bead is in second bacterium
        elif 2 < j <= 5:
            # The distance vector between the two beads from other two bacteria
            r_12 = r_arr_2[i,j-3,] - r_arr_1[i,:]
            r_32 = r_arr_2[i,j-3,] - r_arr_3[i,:]

            # The force experienced due to Van der Waals force
            for k in range(3):
                F_van = van_der_waals_force(A_Ham, r_12[k]) + van_der_waals_force(A_Ham, r_32[k])
                F_van_arr_2[k] = F_van

        # If the bead is in the third bacterium
        elif 5 < j <= 8:
            # The distance vector between the two beads from other two bacteria
            r_13 = r_arr_3[i,j-6,] - r_arr_1[i,:]
            r_23 = r_arr_3[i,j-6,] - r_arr_2[i,:]

            # The force experienced due to Van der Waals force
            for k in range(3):
                F_van = van_der_waals_force(A_Ham, r_13[k]) + van_der_waals_force(A_Ham, r_23[k])
                F_van_arr_3[k] = F_van
        """
        ##print(f"{heading:-^30}")

    F_total = np.zeros((num_beads,3),dtype=np.float64)
    for j in range(0,num_beads,beads_per_bacterium):
        ##print(f"{heading:=^30}")
        # For each bacterium
        F_ex = spring_force(r_arr[i,j+0,],r_arr[i,j+1,],r_arr[i,j+2,]) + bending_force(bond_stiff,r_arr[i,j+0,],r_arr[i,j+1,],r_arr[i,j+2,])
        ##print(F_ex)
        F_total[j:j+3] = F_van_arr[j:j+3] + F_ex
    
    
    for j in range(num_beads):
        a_j = F_total[j]/mass
        # Particle j
        v_j = step1(v_arr[i,j,:], dt, a_j[:])
        delta_vj = step2(v_j, f, random_1[i,j,:])
        r_arr[i+1,j,:] = step3(r_arr[i,j,:], v_j, delta_vj, dt)
        v_arr[i+1,j,:] = step4(v_j, delta_vj)
    """
    # Particle j
    v_j = step1(v_arr_1[i,1,:], dt, a_1[1,:])
    delta_vj = step2(v_j, f, random_1[i,1,:])
    r_arr_1[i+1,1,:] = step3(r_arr_1[i,1,:], v_j, delta_vj, dt)
    v_arr_1[i+1,1,:] = step4(v_j, delta_vj)
    # Particle k
    v_k = step1(v_arr_1[i,2,:], dt, a_1[2,:])
    delta_vk = step2(v_k, f, random_1[i,2,:])
    r_arr_1[i+1,2,:] = step3(r_arr_1[i,2,:], v_k, delta_vk, dt)
    v_arr_1[i+1,2,:] = step4(v_k, delta_vk)
    """
    """
    # For second bacterium
    F_ex_2 = spring_force(r_arr_2[i,0,],r_arr_2[i,1,],r_arr_2[i,2,]) + bending_force(bond_stiff,r_arr_2[i,0,],r_arr_2[i,1,],r_arr_2[i,2,])
    F_total_2 = F_van_arr_2 + F_ex_2
    a_2 = F_total_2/mass
    
    # Particle i
    v_i = step1(v_arr_2[i,0,:], dt, a_2[0,:])
    delta_vi = step2(v_i, f, random_2[i,0,:])
    r_arr_2[i+1,0,:] = step3(r_arr_2[i,0,:], v_i, delta_vi, dt)
    v_arr_2[i+1,0,:] = step4(v_i, delta_vi)
    # Particle j
    v_j = step1(v_arr_2[i,1,:], dt, a_2[1,:])
    delta_vj = step2(v_j, f, random_2[i,1,:])
    r_arr_2[i+1,1,:] = step3(r_arr_2[i,1,:], v_j, delta_vj, dt)
    v_arr_2[i+1,1,:] = step4(v_j, delta_vj)
    # Particle k
    v_k = step1(v_arr_2[i,2,:], dt, a_2[2,:])
    delta_vk = step2(v_k, f, random_2[i,2,:])
    r_arr_1[i+1,2,:] = step3(r_arr_1[i,2,:], v_k, delta_vk, dt)
    v_arr_1[i+1,2,:] = step4(v_k, delta_vk)
    
    # For third bacterium
    F_ex_3 = spring_force(r_arr_3[i,0,],r_arr_3[i,1,],r_arr_3[i,2,]) + bending_force(bond_stiff,r_arr_3[i,0,],r_arr_3[i,1,],r_arr_3[i,2,])
    F_total_3 = F_van_arr_3 + F_ex_3
    a_3 = F_total_2/mass
    
    # Particle i
    v_i = step1(v_arr_3[i,0,:], dt, a_3[0,:])
    delta_vi = step2(v_i, f, random_3[i,0,:])
    r_arr_3[i+1,0,:] = step3(r_arr_3[i,0,:], v_i, delta_vi, dt)
    v_arr_3[i+1,0,:] = step4(v_i, delta_vi)
    # Particle j
    v_j = step1(v_arr_3[i,1,:], dt, a_3[1,:])
    delta_vj = step2(v_j, f, random_3[i,1,:])
    r_arr_3[i+1,1,:] = step3(r_arr_3[i,1,:], v_j, delta_vj, dt)
    v_arr_3[i+1,1,:] = step4(v_j, delta_vj)
    # Particle k
    v_k = step1(v_arr_3[i,2,:], dt, a_3[2,:])
    delta_vk = step2(v_k, f, random_3[i,2,:])
    r_arr_3[i+1,2,:] = step3(r_arr_3[i,2,:], v_k, delta_vk, dt)
    v_arr_3[i+1,2,:] = step4(v_k, delta_vk)
    """
    i += 1
    
# Plot the graph of the path

fig = plt.figure()
 
# syntax for 3-D projection
ax = fig.add_subplot(projection ='3d')

# plotting
ax.plot3D(r_arr[:,0,0], r_arr[:,0,1], r_arr[:,0,2], color = "#f57e21",
          label = "$r_i$ ")
ax.plot3D(r_arr[:,1,0], r_arr[:,1,1], r_arr[:,1,2], color = "#f57e21",
          label = "$r_j$")
ax.plot3D(r_arr[:,2,0], r_arr[:,2,1], r_arr[:,2,2], color = "#f57e21",
          label = "$r_k$")
ax.plot3D(r_arr[:,3,0], r_arr[:,3,1], r_arr[:,3,2], color = "#ffbb00",
          label = "$r_i$ ")
ax.plot3D(r_arr[:,4,0], r_arr[:,4,1], r_arr[:,4,2], color = "#ffbb00",
          label = "$r_j$")
ax.plot3D(r_arr[:,5,0], r_arr[:,5,1], r_arr[:,5,2], color = "#ffbb00",
          label = "$r_k$")
ax.plot3D(r_arr[:,6,0], r_arr[:,6,1], r_arr[:,6,2], color = "#69a040",
          label = "$r_i$ ")
ax.plot3D(r_arr[:,7,0], r_arr[:,7,1], r_arr[:,7,2], color = "#69a040",
          label = "$r_j$")
ax.plot3D(r_arr[:,8,0], r_arr[:,8,1], r_arr[:,8,2], color = "#69a040",
          label = "$r_k$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title('3D line plot for single bacteria')
ax.set_xlim3d(-0.5e-6,4.5e-6)
ax.set_ylim3d(-2.5e-6,2.5e-6)
ax.set_zlim3d(-2.5e-6,2.5e-6)
plt.legend()

plt.show()

# Save the result using Pandas DataFrame
# r_arr_1 = np.reshape(r_arr_1,(num_beads * numpoint,3))
# r_df = pd.DataFrame(r_arr_1)


        
