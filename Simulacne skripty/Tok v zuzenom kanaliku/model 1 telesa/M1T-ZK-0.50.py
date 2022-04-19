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

from boundaries_adjustable import *

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
directory = "output/30.3_thin_canal/sim"+str(sim_id)
os.makedirs(directory)

vtk_directory = directory + "/vtk"
os.makedirs(vtk_directory)

#ARGS ----------------------------------------

# boundary speed
v_top = 0.005
v_bottom = -0.005

# inner particles
phi = 0.5

# channel constants
boxX = 120.0
boxY = 60.0
boxZ = 15.0
width = 2.0

capillary_length = 60.0

# inner particle parameters
particle_mass = 0.5

# particle-particle interactions (DPD + hat)
dpd_gamma = 3.0
r_part = 0.5
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

cell_radius = 5.0

# save oif_classes
shutil.copyfile("src/python/object_in_fluid/oif_classes.py", directory + "/oif_classes.py")

cell_type = oif.OifCellType(nodes_file="input/bicluster1134nodes.dat",
                           triangles_file="input/bicluster1134triangles.dat",
                           check_orientation=False,
                           system=system,
                           ks=0.05,
                           kb=0.1,
                           kal=0.2,
                           kag=0.7,
                           kv=0.9,
                           normal=True,
                           resize=[cell_radius, cell_radius, cell_radius])

cell = oif.OifCell(cell_type=cell_type,
                          particle_type=0,
                          origin=[10.0,boxY/2+1.5,boxZ/2],
                          particle_mass=particle_mass,
                          rotate=[math.pi/2.0,0.0,0.0],
                          exclusion_neighbours=False)

# load cluster
cluster = oif.OifCluster()
cluster.add_cell(cell)

# lennard-jones interactions
cluster.set_lennard_jones_interactions(system=system, lj_eps=lj_eps, lj_sigma=lj_sigma, lj_cutoff=lj_cutoff, lj_shift=0.0)
cluster.set_self_cell_soft_sphere_interactions(system=system)

# # zvysenie tuhosti bunky
# for cell in cluster.cells:
    # cell.cell_type.ks = 0.05
    # cell.cell_type.kb = 0.05
    # cell.cell_type.kal= 0.5
    # cell.cell_type.kag= 1.4
    # cell.cell_type.kv = 2.0
    
# SPLIT
point_left = []
point_right = []

for id, point in enumerate(cell.mesh.points):
    if point.get_pos()[2] - cell.get_origin()[2] <= 0.01:
        point_left.append(point)
    if point.get_pos()[2] - cell.get_origin()[2] >= -0.01:
        point_right.append(point)

print("split_left count " + str(len(point_left)))
print("split_right count " + str(len(point_right)))

cluster.set_rotation([0,0,math.pi/2.0])

# create boundaries
boundaries = []
boundary_particle_type = 10

# cell-wall interactions
cluster.set_cell_boundary_interactions(system, boundary_particle_type, soft_a=0.00022,
                                                                       soft_n=2,
                                                                       soft_cutoff=0.6,
                                                                       soft_offset=0.0)

#self-cell interactions
cluster.set_self_cell_soft_sphere_interactions(system)
                         
print("Cell interactions created")

print("Generating and creating Inner particles")
# generating inner particles
part_volume = 4.0 * np.pi * (r_part ** 3) / 3.0
ncells = len(cluster.cells)

for i, cell in enumerate(cluster.cells):
    n_part = int(phi*cell.volume()/part_volume)

    cell.inner_particles.seed_box(system, part_r=r_part, n=n_part, a=4.5, b=15, c=4.5, particle_type=11 + i, particle_mass=particle_mass,seed=0)
    
    cell.inner_particles.set_interactions(dpd_gamma, hat_fmax, soft_a, soft_n, soft_cutoff=2*r_part)

 
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


boundaries = list()

ax = capillary_length
ay = 9.0
az = 9.0
FillBoundaries(boundaries, vtk_directory, ax, ay, az, boxX, boxY, boxZ, width) 

for boundary in boundaries:
    system.lbboundaries.add(lbboundaries.LBBoundary(shape=boundary))
    system.constraints.add(shape=boundary, particle_type=boundary_particle_type, penetrable=False)

rh_v = 0.0005

left_flow = shapes.Rhomboid(corner=[4.0, -1.0, -1.0], a=[1.0, 0.0, 0.0],
                            b=[0.0, 0.0, boxZ+2], c=[0.0, boxY+2, 0.0], direction=1)
system.lbboundaries.add(lbboundaries.LBBoundary(shape=left_flow, velocity = [rh_v, 0.0, 0.0]))

# main integration loop

maxCycle = 5000
steps_in_one_cycle = 2000

out_file = open(directory + "/rotation_" + str(sim_id) + ".txt", "a")
out_data = "time,x_cell_1,x_cell_2,y_cell_1,y_cell_2,z_cell_1,z_cell_2,cell_1_rotation,cell_2_rotation"
out_file.write(out_data + "\n")
out_file.close()

out_file = open(directory + "/data_" + str(sim_id) + ".txt", "a")
out_data = "time,x_vel_cell_1,y_vel_cell_1,z_vel_cell_1,x_pos_cell_1,y_pos_cell_1,z_pos_cell_1,x_vel_cell_2,y_vel_cell_2,z_vel_cell_2,x_pos_cell_2,y_pos_cell_2,z_pos_cell_2,cluster_x,cluster_y,cluster_z,pos_min_x,pos_max_x,pos_min_y,pos_max_y,pos_min_z,pos_max_z"
out_file.write(out_data + "\n")
out_file.close()

print("saving vtk")

# vtk saving
cluster.output_vtk_cluster(vtk_directory,0)
for cell in cluster.cells:
    cell.inner_particles.output_vtk_pos_folded(num=0, output_directory=vtk_directory)

print("starting simulation")


for cycle in range(1, maxCycle):
    loopStartTime = time.time()
    system.integrator.run(steps=steps_in_one_cycle)
    current_time = cycle * system.time_step * steps_in_one_cycle / 1000  # time in ms
    
    # output vtk of cluster(not including inner particles)
    cluster.output_vtk_cluster(vtk_directory, cycle)
    
    # output vtk of inner particles
    for cell in cluster.cells:
        cell.inner_particles.output_vtk_pos_folded(num=cycle, output_directory=vtk_directory)
    
    f_vel = calc_mean_velocity(int(boxX / 2.0))
    print("fluid x-velocity (mean @ mid capillary): " + str(f_vel[0]))
    
    # calculate and output rotations
    cluster_centroid = cell.get_origin()
    cluster_velocity = cell.get_velocity()
    
    print("cluster center: " + str(cluster_centroid))
    print("cluster velcity: " + str(cluster_velocity))
    
    # SPLIT
    
    left_vel = np.array([0.0, 0.0, 0.0])
    left_pos = np.array([0.0, 0.0, 0.0])
    
    for point in point_left:
        left_vel += point.get_vel()
        left_pos += point.get_pos()
        
    left_vel = left_vel / len(point_left)
    left_pos = left_pos / len(point_left)
    
    right_vel = np.array([0.0, 0.0, 0.0])
    right_pos = np.array([0.0, 0.0, 0.0])
    
    for point in point_right:
        right_vel += point.get_vel()
        right_pos += point.get_pos()
        
    right_vel = right_vel / len(point_right)
    right_pos = right_pos / len(point_right)
    
    print("left point pos " + str(left_pos))
    print("right point pos " + str(right_pos))
    
    print("left point vel " + str(left_vel))
    print("right point vel " + str(right_vel))   
    
    # data => position and velocity
    out_data = str(f_vel[0]) + ","

    out_data += str(left_vel[0]) + "," + str(left_vel[1]) + "," + str(left_vel[2]) + ","
    out_data += str(left_pos[0]) + "," + str(left_pos[1]) + "," + str(left_pos[2]) + ","
    
    out_data += str(right_vel[0]) + "," + str(right_vel[1]) + "," + str(right_vel[2]) + ","
    out_data += str(right_pos[0]) + "," + str(right_pos[1]) + "," + str(right_pos[2]) + ","

    cluster_origin = cluster.get_origin()
    out_data += str(cluster_origin[0]) + "," + str(cluster_origin[1]) + "," + str(cluster_origin[2]) + ","
    
    #elongation
    pos_bounds = cluster.cells[0].pos_bounds()
    out_data += str(pos_bounds[0]) + "," + str(pos_bounds[1]) + "," + str(pos_bounds[2]) + "," + str(pos_bounds[3]) + "," + str(pos_bounds[4]) + "," + str(pos_bounds[5])

    out_file = open(directory + "/data_" + str(sim_id) + ".txt", "a")
    out_file.write(str(current_time) + "," + out_data + "\n")
    out_file.close()
    
    out_data = str(current_time) + ","

    left_rotation_X = (left_vel[0] - cluster_velocity[0]) / (distance(left_pos, cluster_centroid))
    left_rotation_Y = (left_vel[1] - cluster_velocity[1]) / (distance(left_pos, cluster_centroid))
    left_rotation_Z = (left_vel[2] - cluster_velocity[2]) / (distance(left_pos, cluster_centroid))
    
    out_data += "{:.9f},{:.9f},{:.9f},".format(left_rotation_X,left_rotation_Y,left_rotation_Z)
    
    out_data += str(np.sqrt(left_rotation_X * left_rotation_X + left_rotation_Y * left_rotation_Y + left_rotation_Z * left_rotation_Z)) + ","
    
     #rotation right point
    left_rotation_X = (right_vel[0] - cluster_velocity[0]) / (distance(right_pos, cluster_centroid))
    left_rotation_Y = (right_vel[1] - cluster_velocity[1]) / (distance(right_pos, cluster_centroid))
    left_rotation_Z = (right_vel[2] - cluster_velocity[2]) / (distance(right_pos, cluster_centroid))
    
    out_data += "{:.9f},{:.9f},{:.9f},".format(left_rotation_X,left_rotation_Y,left_rotation_Z)
    
    out_data += str(np.sqrt(left_rotation_X * left_rotation_X + left_rotation_Y * left_rotation_Y + left_rotation_Z * left_rotation_Z))
    out_file = open(directory + "/rotation_" + str(sim_id) + ".txt", "a")
    out_file.write(out_data + "\n")
    out_file.close()
    
    if(current_time % 10 == 0):
        cluster.save_cluster(directory=directory + "/" + str(current_time), save_interactions=True, save_inner_particles=True)
    
    
    print("currTime: " + str(current_time))
    loopEndTime = time.time()
    print("\n...........\n...whole loop took " + str(loopEndTime - loopStartTime) + " s\n...........\n")
print ("Simulation completed.")
exit()
    

