import espressomd
import object_in_fluid as oif
import numpy as np

from espressomd import shapes

#when calling in the script, use:
#boundaries = list()
#FillBoundaries(boundaries, vtk_directory, par_ax, par_ay, par_az)
#for boundary in boundaries:
#    system.lbboundaries.add(lbboundaries.LBBoundary(shape=boundary))
#    system.constraints.add(shape=boundary, particle_type=10, penetrable=False)


def FillBoundaries(boundaries, vtk_directory, ax, ay, az, boxX, boxY, boxZ, width):
    # GEOMETRY
    # rectangular channel with size boxX*boxY*boxZ, the width of the borders is set with width
    # of the narrow rectangle part in the middle with cross-section ay*az and length ax, the entrance is smoothed

    # center of the channel
    cx = boxX / 2
    cy = boxY / 2
    cz = boxZ / 2

    # size of obstacles that create the narrow part
    # horizontal
    ah = ax
    bh = 2 * width + boxY
    ch = width + (boxZ - az) / 2
    # vertical
    av = ax
    bv = width + (boxY - ay) / 2
    cv = 2 * width + boxZ

    # corners for obstacles
    # A
    A_x = cx - ax / 2
    A_y = -width
    A_z = -width
    # B
    B_x = cx - ax / 2
    B_y = (boxY + ay) / 2
    B_z = -width
    # C
    C_x = cx - ax / 2
    C_y = -width
    C_z = (boxZ + az) / 2

    # size of obstacles that create 45 degree angle narrowing Left+Right
    odvesna = (boxY - ay)/2 - width
    vyska = boxZ+2*width

    # corners for obstacles that create 45 degree angle narrowing Left+Right
    # R
    R_x = cx - ax/2 - odvesna
    R_y = -width
    R_z = -width
    # L
    L_x = cx - ax / 2 - odvesna
    L_y = boxY + width
    L_z = -width

    # cylinder
    polomer = (boxY - ay)/4

    # cylinder entrance centres
    #left
    Cl_x = cx - ax / 2
    Cl_y = cy + ay/2 + polomer
    Cl_z = cz
    #right
    Cr_x = cx - ax / 2
    Cr_y = cy - ay/2 - polomer
    Cr_z = cz



    wallBottom = shapes.Rhomboid(corner=[-width, -width, -width], a=[boxX+2*width, 0.0, 0.0], b=[0.0, boxY+2*width, 0.0], c=[0.0, 0.0, width], direction=1)
    wallTop = shapes.Rhomboid(corner=[-width, -width, boxZ], a=[boxX+2*width, 0.0, 0.0], b=[0.0, boxY+2*width, 0.0], c=[0.0, 0.0, width], direction=1)
    wallBack = shapes.Rhomboid(corner=[-width, boxY, -width], a=[boxX+2*width, 0.0, 0.0], b=[0.0, width, 0.0], c=[0.0, 0.0, boxZ+2*width], direction=1)
    wallFront = shapes.Rhomboid(corner=[-width, -width, -width], a=[boxX+2*width, 0.0, 0.0], b=[0.0, width, 0.0], c=[0.0, 0.0, boxZ+2*width], direction=1)
    horizBottom = shapes.Rhomboid(corner=[A_x, A_y, A_z], a=[ah, 0, 0],    b=[0, bh, 0], c=[0, 0, ch], direction = 1)
    horizTop = shapes.Rhomboid(corner=[C_x, C_y, C_z], a=[ah, 0, 0],   b=[0, bh, 0], c=[0, 0, ch], direction = 1)
    verticLeft = shapes.Rhomboid(corner=[A_x, A_y, A_z], a=[av, 0, 0], b=[0, bv, 0], c=[0, 0, cv], direction = 1)
    verticRight = shapes.Rhomboid(corner=[B_x, B_y, B_z], a=[av, 0, 0],   b=[0, bv, 0], c=[0, 0, cv], direction = 1)

    entranceLeft = shapes.Rhomboid(corner=[L_x - (np.sqrt(3) / 2)*polomer + width, L_y , L_z], a=[odvesna, -odvesna, 0], b=[odvesna, 0, 0], c=[0, 0, vyska], direction=1)
    entranceRight = shapes.Rhomboid(corner=[R_x - (np.sqrt(3) / 2)*polomer + width, R_y, R_z], a=[odvesna, 0, 0], b=[odvesna, odvesna, 0], c=[0, 0, vyska], direction=1)

    entrCylLeft = shapes.Cylinder(center=[Cl_x, Cl_y, Cl_z], axis=[0, 0, 1], length=boxZ + 2*width, radius=polomer, direction=1)
    entrCylRight = shapes.Cylinder(center=[Cr_x, Cr_y, Cr_z], axis=[0, 0, 1], length=boxZ + 2*width, radius=polomer, direction=1)
   
    exitLeft = shapes.Rhomboid(corner=[L_x + ax + odvesna - width + (np.sqrt(3) / 2)*polomer, L_y , L_z], a=[-odvesna, -odvesna, 0], b=[odvesna, 0, 0], c=[0, 0, vyska], direction=1)
    exitRight = shapes.Rhomboid(corner=[R_x + ax + odvesna - width + (np.sqrt(3) / 2)*polomer, R_y , R_z], a=[odvesna, 0, 0], b=[-odvesna, odvesna, 0], c=[0, 0, vyska], direction=1)

    exitCylLeft = shapes.Cylinder(center=[Cl_x + ax, Cl_y, Cl_z], axis=[0, 0, 1], length=boxZ + 2*width, radius=polomer, direction=1)
    exitCylRight = shapes.Cylinder(center=[Cr_x + ax, Cr_y, Cr_z], axis=[0, 0, 1], length=boxZ + 2*width, radius=polomer, direction=1)


    oif.output_vtk_rhomboid(rhom_shape=wallBottom, out_file=vtk_directory + "/wallBottom.vtk")
    oif.output_vtk_rhomboid(rhom_shape=wallTop, out_file=vtk_directory + "/wallTop.vtk")
    oif.output_vtk_rhomboid(rhom_shape=wallBack, out_file=vtk_directory + "/wallBack.vtk")
    oif.output_vtk_rhomboid(rhom_shape=wallFront, out_file=vtk_directory + "/wallFront.vtk")
    oif.output_vtk_rhomboid(rhom_shape=horizBottom, out_file=vtk_directory + "/horizBottom.vtk")
    oif.output_vtk_rhomboid(rhom_shape=horizTop, out_file=vtk_directory + "/horizTop.vtk")
    oif.output_vtk_rhomboid(rhom_shape=verticLeft, out_file=vtk_directory + "/verticLeft.vtk")
    oif.output_vtk_rhomboid(rhom_shape=verticRight, out_file=vtk_directory + "/verticRight.vtk")

    oif.output_vtk_rhomboid(rhom_shape=entranceLeft, out_file=vtk_directory + "/entranceLeft.vtk")
    oif.output_vtk_rhomboid(rhom_shape=entranceRight, out_file=vtk_directory + "/entranceRight.vtk")

    oif.output_vtk_cylinder(cyl_shape=entrCylLeft, n=20, out_file=vtk_directory + "/entrCylLeft.vtk")
    oif.output_vtk_cylinder(cyl_shape=entrCylRight, n=20, out_file=vtk_directory + "/entrCylRight.vtk")

    oif.output_vtk_rhomboid(rhom_shape=exitLeft, out_file=vtk_directory + "/exitLeft.vtk")
    oif.output_vtk_rhomboid(rhom_shape=exitRight, out_file=vtk_directory + "/exitRight.vtk")

    oif.output_vtk_cylinder(cyl_shape=exitCylLeft, n=20, out_file=vtk_directory + "/exitCylLeft.vtk")
    oif.output_vtk_cylinder(cyl_shape=exitCylRight, n=20, out_file=vtk_directory + "/exitCylRight.vtk")

    
    boundaries.append(wallBottom)
    boundaries.append(wallTop)
    boundaries.append(wallBack)
    boundaries.append(wallFront)
    boundaries.append(horizBottom)
    boundaries.append(horizTop)
    boundaries.append(verticLeft)
    boundaries.append(verticRight)
    boundaries.append(entranceLeft)
    boundaries.append(entranceRight)
    boundaries.append(entrCylLeft)
    boundaries.append(entrCylRight)
    boundaries.append(exitLeft)
    boundaries.append(exitRight)
    boundaries.append(exitCylLeft)
    boundaries.append(exitCylRight)
        
    return 0
