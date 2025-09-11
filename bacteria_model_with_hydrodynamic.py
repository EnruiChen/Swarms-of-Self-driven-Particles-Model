# =============================================================================
# This is the code for modelling three bacteria
# Based on the single bacteria model
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
from sklearn.neighbors import NearestNeighbors

start = time.time()

import matplotlib as mpl

mu, sigma = 0, 0.5 # mean and standard deviation
numpoint = 1000
seed = 144
num_bacteria = 7
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
    The Van der Waals force with repulsive force include.
    
    Parameters
    ----------
    r_ab : distance vector betwen a and b, unit: metre.
    A_Ham : Hamaker constant, unit: Joule.

    Returns
    -------
    Force : The Van der Waals force exert from a to b.

    """
    r_ab_scalar = np.linalg.norm(r_ab)
    Force = (-(A_Ham/12) * (radius/(r_ab_scalar**2 - 4 * radius * r_ab_scalar
                                 + 4 * radius**2)) 
             + 4.5e-10 * np.exp(-5.4e8 * (r_ab_scalar - 2*radius))) * (r_ab/r_ab_scalar)
    force_cap = 1e-13 # Newtons, maximum force allowed
    force_magnitude = np.linalg.norm(Force)
    if force_magnitude > force_cap:
        Force = Force * force_cap/force_magnitude
    return Force

# Define multivariable Gaussian distribution function
def random_multivar(mean,cov):
    """
    
    Parameters
    ----------
    mean : array
        The mean value of displacement. 
    cov : array
        The covariance of displacement.

    Returns
    -------
    random_displacement : array
        random displacement sample in multivariable normal distribution

    """
    random_displacement = rng.multivariate_normal(mean, cov)
    random_displacement = random_displacement.reshape(num_beads,3)

    return random_displacement

def diffusion_matrix(r_ij_arr):
    """
    
    Parameters
    ----------
    r_ij_arr : array size=[num_beads,num_beads,3]
        displacement of the two beads.
        we need to calculate the ri - rj

    Returns
    -------
    D_array : array
        diffusion matrix.

    """
    D_array = np.zeros((num_beads,num_beads),dtype=object)
    for i in range(num_beads):
        for j in range(num_beads):
            # Set value that will use in the loop
            x_ij = r_ij_arr[i,0] - r_ij_arr[j,0]
            y_ij = r_ij_arr[i,1] - r_ij_arr[j,1]
            z_ij = r_ij_arr[i,2] - r_ij_arr[j,2]
            r_ij = np.linalg.norm(r_ij_arr[i] - r_ij_arr[j])

            if i == j:
                # Identity matrix if i=j
                D_array[i,j] = k_B * T/(
                    6*np.pi*viscosity*radius) * np.identity(3)
            else:
                D_array[i,j] = k_B * T/(8*np.pi*viscosity*r_ij**3) * np.array([
                    [r_ij**2+x_ij**2, x_ij*y_ij, x_ij*z_ij],
                    [y_ij*x_ij, r_ij**2+y_ij**2, y_ij*z_ij],
                    [z_ij*x_ij, z_ij*y_ij, r_ij**2 + z_ij**2]])

    return D_array

def driving_force(force, rj, rk):
    """
    

    Parameters
    ----------
    force : float
        magnitude of the driving force.
    rj : array
        position of the first bead.
    rk : array
        position of the last bead.

    Returns
    -------
    driving_force : array
        driving force acting on all beads.

    """
    vector = rk - rj
    
    # Calculate the magnitude of the vector
    magnitude = np.linalg.norm(vector)
    
    # Calcualte the normalized direction vector
    normalize_vector = vector / magnitude
    
    # Times the direction vector with magnitude of the driving force
    driving_force = normalize_vector * force
    return driving_force
    
def get_order(arr,nmax):
    """
    Function to calculate the order parameter of bactria using the Q tensor approach.
    Function returns the maximum value of the eigenvalue which tells us how good
    the orientation is.

    Parameters
    ----------
    arr : array
        Position of the bacteria.
    nmax : int
        side length.

    Returns
    -------
    eigenvalues.max()
        order parameter for bacteria.

    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax)
    lab = np.zeros((3,nmax))
    norm = np.linalg.norm(arr,axis=1)
    for j in range(3):
        lab[j,:] = arr[:,j]/norm[:]
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                Qab[a,b] += 3*lab[a,i]*lab[b,i] - delta[a,b]
    Qab = Qab/(2*nmax)
    eigenvalues, eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()

# Define initial values and arrays to hold values
bond_stiff = 1e-17 # Bond stiffness
mass = 1e-15
viscosity = 1e-3 # viscosity of water in 20 Celseius, Pa s
drag_coefficient = 6 * np.pi * radius * viscosity # Dragg coefficient
dt = 1e-4 # Time step
f = 1 - np.exp(-drag_coefficient * dt/mass)
k_B = 1.38e-23 # Boltzmann constant
T_celsius = 20 # Temperature, unit: Celsius
T = 273.15 + T_celsius # Temperature, unit: K
A_Ham = 12.6e-21 # Hamaker constant for proteins, unit: Joule
driving_scalar = 1.5 * 1e-13

# Array to hold position value for three bacteria with three beads
r_arr = np.zeros((numpoint, num_beads, 3))
# Array to hold velocity value for three bacteria with three beads
v_arr = np.zeros((numpoint, num_beads, 3))
# Array to hold distance with the nearest bacterium
neighbour_arr = np.zeros((numpoint,num_bacteria))
# Array to hold order parameter
order_arr = np.zeros(numpoint)

# Use for loop to create starting position for each bacterium
n = 0
r_z_arr = 0.5 * 1.2 * np.array([-np.sqrt(3),0,np.sqrt(3)])
for r_z in np.repeat(r_z_arr,1):
    if 6 <= n < 15:
        for r_y in 0.5 * 1.2 * np.array([-2,0,2]):
            ri = np.array([1.5, r_y, r_z]) * 1e-6
            rj = np.array([0, r_y, r_z]) * 1e-6
            rk = np.array([3, r_y, r_z]) * 1e-6
            r_arr[0,n:n+3] = np.array([ri,rj,rk])
            n += 3
    else:
        for r_y in 0.5 * 1.2 * np.array([-1,1]):
            ri = np.array([1.5, r_y, r_z]) * 1e-6
            rj = np.array([0, r_y, r_z]) * 1e-6
            rk = np.array([3, r_y, r_z]) * 1e-6
            r_arr[0,n:n+3] = np.array([ri,rj,rk])
            n += 3

# Create random starting position of bacteria
# for r_y, r_z in 15*rng.random((num_bacteria,2)) - 7.5:
#     ri = np.array([1.5, r_y, r_z]) * 1e-6
#     rj = np.array([0, r_y, r_z]) * 1e-6
#     rk = np.array([3, r_y, r_z]) * 1e-6
#     r_arr[0,n:n+3] = np.array([ri,rj,rk])
#     n += 3

n_loop = 0

while n_loop < numpoint - 1:
    # Calculate the force and acceleration
    # For Van der Waals force:
    # Define empty array for storing Van der Waals force
    F_van_arr = np.zeros((num_beads,3))
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
            # The distance vector between the two beads from other two bacteria
                r_jk = r_arr[n_loop,j,] - r_arr[n_loop,k,]
             # The force experienced due to Van der Waals force from each beads 
            #for k in range(3):
                F_van = -van_der_waals_force(A_Ham, r_jk) #+ van_der_waals_force(A_Ham, r_31[k])
            F_van_arr[j] += F_van

    F_total = np.zeros((num_beads,3),dtype=np.float64)
    for j in range(0,num_beads,beads_per_bacterium):
        # For each bacterium
        F_ex = spring_force(r_arr[n_loop,j+0,],r_arr[n_loop,j+1,],r_arr[n_loop,j+2,]) + bending_force(bond_stiff,r_arr[n_loop,j+0,],r_arr[n_loop,j+1,],r_arr[n_loop,j+2,])
        F_total[j:j+3] = F_van_arr[j:j+3] + F_ex + driving_force(driving_scalar, r_arr[n_loop,j+1], r_arr[n_loop,j+2])

    dm = diffusion_matrix(r_arr[n_loop,:])
    dmflat = np.zeros((3*num_beads, 3*num_beads))
    for i in range(num_beads):
        for j in range(num_beads):
            for m in range(3):
                for n in range(3):
                    dmflat[i*3+m, j*3+n] = dm[i,j][m][n]
    forceflat = F_total.ravel().T
    # Change in position:
    r_arr_change = (np.dot(dmflat, forceflat) * dt)/(k_B*T)
    r_arr_change = r_arr_change.reshape(num_beads, 3)

    # Update the positon value
    r_arr[n_loop+1,:] = r_arr[n_loop,:] + r_arr_change + random_multivar(np.zeros(3*num_beads), 2*dmflat*dt)

    # Find nearest neighbour for each bacterium where we take the middle bead position
    r_middle_arr = np.zeros((num_bacteria,3))
    direction_arr = np.zeros((num_bacteria,3)) # array holding direction
    n = 0
    for i in range(num_bacteria):
        r_middle_arr[i] = r_arr[n_loop,n]
        direction_arr[i] = r_arr[n_loop,n+2] - r_arr[n_loop,n+1]
        n += 3
    
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(
        r_middle_arr)
    distances, indices = nbrs.kneighbors(r_middle_arr)
    neighbour_arr[n_loop] = distances[:,1]
    
    # Calculate the order parameter
    order_parameter = get_order(direction_arr, num_bacteria)
    order_arr[n_loop] = order_parameter

    n_loop += 1

neighbour_arr[n_loop] = distances[:,1]
order_arr[n_loop] = order_parameter

end = time.time()

# Plot the graph of the path

cmap = mpl.colormaps["ocean"]
colors = cmap(np.linspace(0, 0.9, num_beads))

fig = plt.figure()
 
# syntax for 3-D projection
ax = fig.add_subplot(projection ='3d')

# plotting
for i in range(num_beads):
    ax.plot3D(r_arr[:,i,0], r_arr[:,i,1], r_arr[:,i,2], color = colors[i])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.set_title(f'3D line plot for {num_bacteria} bacteria with hydrodynamics')
ax.set_xlim3d(-1.5e-6,1e-5)
ax.set_ylim3d(-2e-6,2e-6)
ax.set_zlim3d(-2e-6,2e-6)
##plt.legend()

x =  datetime.datetime.now()
date = x.strftime("%d%m%y-%H%M")
# plt.savefig(f"../picture/{num_bacteria} bacteria with driving force {date}", dpi=100)
plt.show()

total_time = end-start
print(f"{total_time:.3}")
# Save the result using Pandas DataFrame
# r_arr = np.reshape(r_arr[:,0,],(numpoint,3))
# r_df = pd.DataFrame(r_arr)
neighbour_df = pd.DataFrame(np.average(neighbour_arr,axis=1))
order_parameter_df = pd.DataFrame(order_arr)

# r_df.to_csv(f"rj_array_{seed}.csv",index=False,header=False)
neighbour_df.to_csv(f"../data/nearest neighbour/nearest_neighbour_hexagonal_{T_celsius}.csv",
                   index=False,header=False)
order_parameter_df.to_csv(f"../data/order parameter/order_parameter_hexagonal_{T_celsius}.csv",
                          index=False,header=False)

# fig, ax = plt.subplots()

# cmap = mpl.colormaps["gist_ncar"]
# colors = cmap(np.linspace(0, 0.9, num_bacteria))

# for i in range(num_bacteria):
#     if np.average(neighbour_arr[:,i])< 3e-6:
#         ax.hist(neighbour_arr[:,i], bins=20, density=True, alpha = 0.3,
#                 color = colors[i], label=f"{i+1}")
# ax.set_title("The histogram of distance with first nearest neighbour")
# ax.set_xlabel("Distance, (m)")
# plt.legend(title="Bacteria number")
# plt.savefig(f"../picture/nine_bacteria/histogram of {num_bacteria} bacteria {date}", dpi=100)
# plt.show()

# fig, ax = plt.subplots()

# cmap = mpl.colormaps["gist_ncar"]
# colors = cmap(np.linspace(0, 0.9, num_bacteria))

time_arr = np.arange(0,numpoint*dt, dt)

# ax.plot(time_arr[1::],np.average(neighbour_arr[1:,],axis=1))
# ax.set_xlabel("Time, (s)")
# ax.set_ylabel("Distance, (m)")
# ax.set_title(f"The average distance with nearest neighbour with respect to time at {T_celsius}$^\circ$C")
# plt.savefig(f"../picture/nine_bacteria/{T_celsius} Celsius/average distance with nearest neighbour {num_bacteria} bacteria {date}", dpi=100)
# plt.show()

# fig, ax = plt.subplots()

# ax.plot(time_arr[1:], order_arr[1:])
# ax.set_xlabel("Time, (s)")
# ax.set_ylabel("Order parameter, $S$")
# ax.set_title(f"The orderparameter with respect to time at {T_celsius}$^\circ$C")
# plt.savefig(f"../picture/nine_bacteria/{T_celsius} Celsius/order parameter {date}", dpi=100)
# plt.show()

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time, (s)')
ax1.set_ylabel('Distance, (m)', color=color)
ax1.set_title(f"The order parameter and distance with nearest neighbour plot at {T_celsius}$^\circ$C")
ax1.plot(time_arr,np.average(neighbour_arr,axis=1), color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel("Order parameter, $S$", color=color)  # we already handled the x-label with ax1
ax2.plot(time_arr, order_arr, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(f"../picture/{num_bacteria} bacteria with {T_celsius} Celsius order parameter {date}", dpi=100)
plt.show()
