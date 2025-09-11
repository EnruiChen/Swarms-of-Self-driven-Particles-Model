# =============================================================================
# This the code for modeling bacteria
# We first try to model a single bacteria
# The unit of length is micrometre
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.animation as animation

mu, sigma = 0, 0.5 # mean and standard deviation
numpoint = 1000
seed = 144
rng = np.random.default_rng(seed) # random number generator
s = rng.normal(mu, sigma, size = (numpoint,3)) # Use normal distribution in number generator
i_random = s
j_random = rng.normal(mu, sigma, size = (numpoint,3))
k_random = rng.normal(mu, sigma, size = (numpoint,3))

radius = 0.5e-6 # Radius of the three circle represent bacteria
length = 4e-6
l_spring = (length - 2*radius)/2 # Calculate how long is the spring
k = 1e-5 # N/m, spring constant

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
    F_ij = - (np.linalg.norm(x1) - l_spring) * k * (x1/np.linalg.norm(x1))
    F_ji = (np.linalg.norm(x1) - l_spring) * k * (x1/np.linalg.norm(x1))
    F_ki = - (np.linalg.norm(x2) - l_spring) * k * (x2/np.linalg.norm(x2))
    F_ik = (np.linalg.norm(x2) - l_spring) * k * (x2/np.linalg.norm(x2))
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

def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        line.set_data_3d(walk[:num, :].T)
    return lines

# Define initial values and arrays to hold values
bond_stiff = 1e-17 # Bond stiffness
mass = 1e-15
viscosity = 1e-3 # viscosity of water in 20 Celseius, Pa s
drag_coefficient = 6 * np.pi * radius * viscosity # Dragg coefficient
dt = 1e-5 # Time step
f = 1 - np.exp(-drag_coefficient * dt/mass)
k_B = 1.38e-23 # Boltzmann constant
T = 293.15 # Temperature, unit: K

ri_arr = np.zeros((numpoint, 3)) # Array to hold position value
rj_arr = np.zeros((numpoint, 3))
rk_arr = np.zeros((numpoint, 3))
vi_arr = np.zeros((numpoint, 3)) # Array to hold velocity value
vj_arr = np.zeros((numpoint, 3))
vk_arr = np.zeros((numpoint, 3))

ri = np.array([1.5,0,0]) * 1e-6
rj = np.array([0,0,0]) * 1e-6
rk = np.array([3,0,0]) * 1e-6

ri_arr[0,] = ri
rj_arr[0,] = rj
rk_arr[0,] = rk

i = 0

while i < numpoint - 1:
    # Calculate the force and acceleration
    F_ex = spring_force(ri_arr[i],rj_arr[i],rk_arr[i]) + bending_force(bond_stiff,ri_arr[i],rj_arr[i],rk_arr[i])
    a = F_ex/mass

    # Particle i
    v_i = step1(vi_arr[i,:], dt, a[0,:])
    delta_vi = step2(v_i, f, i_random[i,:])
    ri_arr[i+1,:] = step3(ri_arr[i,:], v_i, delta_vi, dt)
    vi_arr[i+1,:] = step4(v_i, delta_vi)
    # Particle j
    v_j = step1(vj_arr[i,:], dt, a[1,:])
    delta_vj = step2(v_j, f, j_random[i,:])
    rj_arr[i+1,:] = step3(rj_arr[i,:], v_j, delta_vj, dt)
    vj_arr[i+1,:] = step4(v_j, delta_vj)
    # Particle k
    v_k = step1(vk_arr[i,:], dt, a[2,:])
    delta_vk = step2(v_k, f, k_random[i,:])
    rk_arr[i+1,:] = step3(rk_arr[i,:], v_k, delta_vk, dt)
    vk_arr[i+1,:] = step4(v_k, delta_vk)
    
    i += 1

print(ri_arr[1:])
print(rj_arr[1:])
print(rk_arr[1:])

# Plot the graph of the path

fig = plt.figure()
 
# syntax for 3-D projection
ax = fig.add_subplot(projection ='3d')

# plotting
ax.plot3D(ri_arr[:,0], ri_arr[:,1], ri_arr[:,2], color = "#f57e21",
          label = "$r_i$ ")
ax.plot3D(rj_arr[:,0], rj_arr[:,1], rj_arr[:,2], color = "#ffbb00",
          label = "$r_j$")
ax.plot3D(rk_arr[:,0], rk_arr[:,1], rk_arr[:,2], color = "#69a040",
          label = "$r_k$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title('3D line plot for single bacterium')
ax.set_xlim3d(-0.5e-6,4.5e-6)
ax.set_ylim3d(-2.5e-6,2.5e-6)
ax.set_zlim3d(-2.5e-6,2.5e-6)
plt.legend()

plt.show()
# fig.savefig("../picture/Single bacterium plot leap-frog method",dpi=100)

# Plot the graph in x-y plane.
fig, ax = plt.subplots()
ax.set_xlabel('x-axis (m)')
ax.set_ylabel('y-axis (m)')

ax.plot(ri_arr[:,0], ri_arr[:,1], color = "#f57e21", label=f"Middle bead")
ax.plot(rj_arr[:,0], rj_arr[:,1], color = "#ffbb00", label=f"First bead")
ax.plot(rk_arr[:,0], rk_arr[:,1], color = "#69a040", label=f"Last bead")
ax.set_xlim(-0.5e-6,4e-6)
ax.set_ylim(-2.5e-6,2.5e-6)
plt.legend()

# fig.savefig("Single bacterium plot leap-frog method x-y plane",dpi=100)

# Plot the graph in y-z plane.
fig, ax = plt.subplots()
ax.set_xlabel('y-axis (m)')
ax.set_ylabel('z-axis (m)')

ax.plot(ri_arr[:,1], ri_arr[:,2], color = "#f57e21", label=f"Middle bead")
ax.plot(rj_arr[:,1], rj_arr[:,2], color = "#ffbb00", label=f"First bead")
ax.plot(rk_arr[:,1], rk_arr[:,2], color = "#69a040", label=f"Last bead")
ax.set_xlim(-2.5e-6,2.5e-6) # Here it is the limit for y-axis
ax.set_ylim(-2.5e-6,2.5e-6) # Here it is the limit for z-axis
plt.legend()

# fig.savefig("Single bacterium plot leap-frog method y-z plane",dpi=100)

# Plot the graph in x-z plane.
fig, ax = plt.subplots()
ax.set_xlabel('x-axis (m)')
ax.set_ylabel('z-axis (m)')

ax.plot(ri_arr[:,0], ri_arr[:,2], color = "#f57e21", label=f"Middle bead")
ax.plot(rj_arr[:,0], rj_arr[:,2], color = "#ffbb00", label=f"First bead")
ax.plot(rk_arr[:,0], rk_arr[:,2], color = "#69a040", label=f"Last bead")
ax.set_xlim(-0.5e-6,4e-6) # Here it is the limit for x-axis
ax.set_ylim(-2.5e-6,2.5e-6) # Here it is the limit for z-axis
plt.legend()

# fig.savefig("Single bacterium plot leap-frog method x-z plane",dpi=100)

plt.show()
##print(Fi_random[:5])
##print(Fj_random[:5])

ri_df = pd.DataFrame(ri_arr)
rj_df = pd.DataFrame(rj_arr)
rk_df = pd.DataFrame(rk_arr)

# Save the number in the array to the CSV file
# ri_df.to_csv("ri_array.csv",index=False,header=False)
# rj_df.to_csv("rj_array.csv",index=False,header=False)
# rj_df.to_csv("rk_array.csv",index=False,header=False)

