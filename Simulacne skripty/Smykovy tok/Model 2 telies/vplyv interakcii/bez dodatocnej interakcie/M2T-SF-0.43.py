import espressomd
import object_in_fluid as oif

from espressomd import lb
from espressomd import lbboundaries
from espressomd import shapes
from espressomd import interactions

import math

import numpy as np
import os, glob, sys, shutil
import time
import random as rnd

from boundaries_adjustable_shear import *

# LL, November 2021

# Adding of Shear-Flow 

# Adjusted morse and soft-sphere interactions between cells to avoid disruption
# Added soft-sphere interactions between particles of the same cell with adjusted parameters 

def distance(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def calc_mean_velocity(x):
    # takes circular cross-section at position x
    # and calculates mean fluid velocity over this cross-section
    ymin = 0
    ymax = int(boxY)
    zmin = 0
    zmax = int(boxZ)
    sum = 0
    counter = 0

    for i in range(ymin, ymax):
        for j in range(zmin, zmax):
            vel = lbf[x,i,j].velocity
            if lbf[x,i,j].boundary == 0:
                sum += vel
                counter += 1
    return sum/counter

def get_position_in_block_random(xo, yo, zo, a, b, c, n):
    #print("xo: " + str(xo) + " yo: " + str(yo) + " zo: " + str(zo))
    #print("a: " + str(a) + " b: " + str(b) + " c:" + str(c))

    ret = []
    for i in range(n):
        ret.append([rnd.uniform(xo, xo+a), rnd.uniform(yo, yo+b), rnd.uniform(zo, zo+c)])
    return ret


if len(sys.argv) != 3:
    print ("1 argument are expected:")
    print ("sim_id: id of the simulation")
    print (" ")

# list of expected arguments
sim_id = "ND"

# read arguments
i = 0
for i, arg in enumerate(sys.argv):
    if i%2 == 1:
        print (str(arg) + " \t" + sys.argv[i + 1])
    if arg == "sim_id":
        sim_id = sys.argv[i + 1]

# check that we have everything
if sim_id == "ND":
    print("something wrong when reading arguments, quitting.")

# create folder structure
directory = "output/18.3/sim"+str(sim_id)
os.makedirs(directory)

vtk_directory = directory + "/vtk"
os.makedirs(vtk_directory)

#ARGS ----------------------------------------

# boundary speed
v_top = 0.005
v_bottom = -0.005

# inner particles
phi = 0.43

# channel constants
boxX = 60.0
boxY = 20.0
boxZ = 50.0
width = 2.0

# inner particle parameters
particle_mass = 1.0

# particle-particle interactions (DPD + hat)
dpd_gamma = 3.0
r_part = 0.3
dpd_cutoff = 2*r_part
hat_fmax = 0.3

# particle-membrane interactions (soft sphere)
soft_a = 0.128
soft_n = 1.5
soft_cutoff = 2*r_part

# cell-cell intractions (Lennard-Jones)
lj_eps = 0.0005  # 0.005
lj_rmin = 0.1   #0.35
lj_sigma = lj_rmin/(2**(1/6))
lj_cutoff = 0.25  #0.3

# system constants
system = espressomd.System(box_l=[boxX, boxY, boxZ])
system.cell_system.skin = 0.2
system.time_step = 0.1

# save script and arguments
shutil.copyfile(str(sys.argv[0]), directory + "/" + str(sys.argv[0]))
out_file = open(directory + "/parameters.txt", "a")
for arg in sys.argv:
    out_file.write(str(arg) + " ")
out_file.write("\n")
out_file.close()

# save oif_classes
shutil.copyfile("src/python/object_in_fluid/oif_classes.py", directory + "/oif_classes.py")

# load cluster
cluster = oif.OifCluster()

# path to save directory which contains nodes, triangles folders and data.json
cluster.load_cluster(system=system, directory='cluster_collection/BiCluster_lj/save_60',origin=[boxX/2.0,boxY/2.0,boxZ/2.0], load_interactions=False)

# rotate 90 degrees 
cluster.set_rotation([0,math.pi / 2.0,0])

cell_radius = cluster.cells[0].cell_type.resize[0]

# lennard-jones interactions
cluster.set_lennard_jones_interactions(system=system, lj_eps=lj_eps, lj_sigma=lj_sigma, lj_cutoff=lj_cutoff, lj_shift=0.0)



# number of nodes in cell mesh
nnode = 1524

# create boundaries
boundaries = []
boundary_particle_type = 10

# cell-wall interactions
cluster.set_cell_boundary_interactions(system, boundary_particle_type, soft_a=0.00022,
                                                                       soft_n=2,
                                                                       soft_cutoff=0.6,
                                                                       soft_offset=0.0)
                         
print("Cell interactions created")

inner_particles = [[],[]]
inner_particle_types = [11, 12]
inner_particle_counts = [0, 0]
inner_positions = [[],[]]

print("Generating Inner particles")
# generating inner particle positions
part_volume = 4.0 * np.pi * (r_part ** 3) / 3.0

for i, cell in enumerate(cluster.cells):
    inner_particle_counts[i] = int(phi * cell.volume() / part_volume)
    print(str(inner_particle_counts[i]))
    origin = cluster.cells[i].get_origin()
    
    n_part = int(phi*cell.volume()/part_volume)
    cell.inner_particles.seed_sphere(system, r_part, n_part, 11 + i, particle_mass=particle_mass, seed=0)
    
    # inner_positions[i] = cell.get_position_in_sphere(r_part, inner_particle_counts[i])
    
    # inner_positions[i] = get_position_in_block_random(origin[0] - inner_radius,
                                                      # origin[1] - inner_radius,
                                                      # origin[2] - inner_radius,
                                                      # 2*inner_radius,
                                                      # 2*inner_radius,
                                                      # 2*inner_radius,
                                                      # inner_particle_counts[i])
    
ncells = len(cluster.cells)
print("Creating inner particles")
# creating inner particles


# interactions between inner particles and membranes
for i in range(ncells):
    system.non_bonded_inter[cluster.cells[i].particle_type, inner_particle_types[i]].soft_sphere.set_params(a=soft_a,
                                                                    n=soft_n,
                                                                    cutoff=soft_cutoff,
                                                                    offset=0.0)

# interactions among inner particles
for i in range(ncells):
    system.non_bonded_inter[inner_particle_types[i], inner_particle_types[i]].hat.set_params(F_max=hat_fmax,
                                                                                            cutoff=dpd_cutoff)
    system.non_bonded_inter[inner_particle_types[i], inner_particle_types[i]].dpd.set_params(weight_function=0,
                                                                     gamma=dpd_gamma,
                                                                     r_cut=dpd_cutoff,
                                                                     trans_weight_function=0,
                                                                     trans_gamma=dpd_gamma,
                                                                     trans_r_cut=dpd_cutoff)
print("Interactions for inner particles created")



FillBoundaries(boundaries, vtk_directory, boxX, boxY, boxZ, width)
 
# fluid 
fluid_viscosity = 1.5
fluid_density = 1.0
lbf = espressomd.lb.LBFluid(agrid=1,
                            dens=fluid_density,
                            visc=fluid_viscosity,
                            tau=system.time_step
                            )
                            
system.actors.add(lbf)
gammaFriction = cluster.cells[0].cell_type.suggest_LBgamma(visc = fluid_viscosity, dens = fluid_density)
system.thermostat.set_lb(LB_fluid=lbf,
                         gamma=gammaFriction)
 

for boundary in boundaries:
  system.constraints.add(shape=boundary, particle_type=boundary_particle_type, penetrable=False)

system.lbboundaries.add(lbboundaries.LBBoundary(shape=boundaries[0],
                                                velocity=[v_bottom, 0.0, 0.0]))
system.lbboundaries.add(lbboundaries.LBBoundary(shape=boundaries[1],
                                                velocity=[v_top, 0.0, 0.0]))

# main integration loop

maxCycle = 5000
steps_in_one_cycle = 2000

out_file = open(directory + "/rotation_" + str(sim_id) + ".txt", "a")
out_data = "time,x_cell_1,x_cell_2,y_cell_1,y_cell_2,z_cell_1,z_cell_2,cell_1_rotation,cell_2_rotation"
out_file.write(out_data + "\n")
out_file.close()


cluster.output_vtk_cluster(vtk_directory, 0)
for cycle in range(1, maxCycle):
    loopStartTime = time.time()
    system.integrator.run(steps=steps_in_one_cycle)
    current_time = cycle * system.time_step * steps_in_one_cycle / 1000  # time in ms
    
    # vtk 
    cluster.output_vtk_cluster(vtk_directory, cycle)
    #for id, cell in enumerate(cells):
    for id in range(ncells):
        system.part.writevtk(
            (vtk_directory + "/inner" + str(inner_particle_types[id]) + "_sim" + str(sim_id) + "_" + str(cycle) + ".vtk"),
            types=[(inner_particle_types[id])])
    
        f_vel = calc_mean_velocity(int(boxX / 2.0))
    print("fluid x-velocity (mean @ mid capillary): " + str(f_vel[0]))

    velocities = []
    positions = []
    out_data = str(current_time) + " " + str(f_vel[0]) + " "
    for id, cell in enumerate(cluster.cells):
        velocities.append(cell.get_velocity())
        positions.append(cell.get_origin())
        print("cell" + str(id) + " velocity: " + str(velocities[id]))
        out_data += str(velocities[id][0]) + " " + str(velocities[id][1]) + " " + str(velocities[id][2]) + " "
        out_data += str(positions[id][0]) + " " + str(positions[id][1]) + " " + str(positions[id][2]) + " "

    out_file = open(directory + "/data_" + str(sim_id) + ".txt", "a")
    out_file.write(str(current_time) + " " + out_data + "\n")
    out_file.close()

    # calculate and output rotations
    cluster_centroid = sum(positions) / ncells
    cluster_velocity = sum(velocities) / ncells
    rotationX = []
    rotationY = []
    rotationZ = []
    out_data = str(current_time) + ","
    for id, cell in enumerate(cluster.cells):
        rotationX.append((velocities[id][0] - cluster_velocity[0]) / (distance(positions[id], cluster_centroid)))
        rotationY.append((velocities[id][1] - cluster_velocity[1]) / (distance(positions[id], cluster_centroid)))
        rotationZ.append((velocities[id][2] - cluster_velocity[2]) / (distance(positions[id], cluster_centroid)))
    for r in rotationX:
        out_data += "{:.9f},".format(r)
        #out_data += str(r) + ","
    for r in rotationY:
        out_data += "{:.9f},".format(r)
        #out_data += str(r) + ","
    for r in rotationZ:
        out_data += "{:.9f},".format(r) 
        #out_data += str(r) + ","
    for i in range(len(rotationX)):
        out_data += str(np.sqrt(rotationX[i] * rotationX[i] + rotationY[i] * rotationY[i] + rotationZ[i] * rotationZ[i])) + ","
    out_file = open(directory + "/rotation_" + str(sim_id) + ".txt", "a")
    out_file.write(out_data + "\n")
    out_file.close()
    
    
    print("currTime: " + str(current_time))
    loopEndTime = time.time()
    print("\n...........\n...whole loop took " + str(loopEndTime - loopStartTime) + " s\n...........\n")
   
print ("Simulation completed.")
exit()
    

