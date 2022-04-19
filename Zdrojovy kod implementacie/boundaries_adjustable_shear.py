import espressomd
import object_in_fluid as oif
import numpy as np

from espressomd import shapes

def FillBoundaries(boundaries, vtk_directory, boxX, boxY, boxZ, width):
    # GEOMETRY
    # basic rectangular channel with size boxX*boxY*boxZ

    wallBottom = shapes.Rhomboid(corner=[-width, -width, -width], a=[boxX+2*width, 0.0, 0.0], b=[0.0, boxY+2*width, 0.0], c=[0.0, 0.0, width], direction=1)
    wallTop = shapes.Rhomboid(corner=[-width, -width, boxZ], a=[boxX+2*width, 0.0, 0.0], b=[0.0, boxY+2*width, 0.0], c=[0.0, 0.0, width], direction=1)

    oif.output_vtk_rhomboid(rhom_shape=wallBottom, out_file=vtk_directory + "/wallBottom.vtk")
    oif.output_vtk_rhomboid(rhom_shape=wallTop, out_file=vtk_directory + "/wallTop.vtk")

    boundaries.append(wallBottom)
    boundaries.append(wallTop)

    return 0
