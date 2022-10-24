import numpy as np
import math as m
from sympy import *


# Generate rotation matrices around each axes on certain angle
# Since rotation around x and y axis is only by pi/2, then cos(angle) is always 0 and sin(angle) is either 1 or -1
# sign is either 1 (for counter clockwise rotation) or -1 (for clockwise rotation)
def Rx(sign):
    return np.array([[1, 0, 0, 0],
                     [0, 0, -sign, 0],
                     [0, sign, 0, 0],
                     [0, 0, 0, 1]])


def Ry(sign):
    return np.array([[0, 0, sign, 0],
                     [0, 1, 0, 0],
                     [-sign, 0, 0, 0],
                     [0, 0, 0, 1]])


def Rz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                     [np.sin(angle), np.cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


# Generate symbolic rotation matrices around each axes
def Rz_symb(angle):
    return np.array([[cos(angle), -sin(angle), 0, 0],
                     [sin(angle), cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


# Generate translation matrices along each axes on certain distance
def Tx(d):
    return np.array([[1, 0, 0, d],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def Ty(d):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, d],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def Tz(d):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, d],
                     [0, 0, 0, 1]])


# Translation to each frame from previous
def T01(q, d):
    return Rz(q[0]) @ Tz(d[0]) @ Rx(1)


def T12(q, d):
    return Rz(q[1]) @ Tx(d[1])


def T23(q, d):
    return Rz(q[2]) @ Tx(d[2]) @ Ry(-1)


def T34(q, d):
    return Rz(q[3]) @ Tz(d[3]) @ Rx(-1)


def T45(q, d):
    return Rz(q[4]) @ Ty(d[4]) @ Rx(1)


def T56(q, d):
    return Rz(q[5]) @ Tz(d[5])


# Solve forward kinematics
def FK_solve(q, d, trans, flag="ee"):
    trans_matrices = [trans,
                      T01(q, d),
                      T12(q, d),
                      T23(q, d),
                      T34(q, d),
                      T45(q, d),
                      T56(q, d)]

    transformation = np.identity(4)
    frames = []
    for i in range(len(trans_matrices)):
        transformation = transformation @ trans_matrices[i]
        frames.append(transformation)

    print(f"\nTransformation matrix from frame 0 to frame 6 is\n{frames[-1]}\n")
    # Return either all frames or only final matrix
    if flag == "ee":
        return [frames[-1]]
    elif flag == "full":
        return frames


def transform_base(trans, q=None, flag="ee", d=None):
    if d is None:
        d = [1, 2, 3, 4, 5, 6]
    if q is None:
        q = [np.pi / 3, np.pi / 3, np.pi / 2, np.pi / 2, np.pi / 6, np.pi / 6]
    frames = FK_solve(q, d, trans, flag)
    return frames


# Solve inverse kinematics
def IK_solve(ee_frame, base_frame=np.identity(4), d=None):
    if d is None:
        d = [1, 2, 3, 4, 5, 6]

    Pc = ee_frame[-1] @ np.linalg.inv(T56(q, d)) @ np.linalg.inv(T45(q, d)) @ np.linalg.inv(T34(q, d))
    xc, yc, zc = Pc[0, 3], Pc[1, 3], Pc[2, 3]

    q1_sol = m.atan(yc / xc)

    q1, q2, q3, q4, q5, q6 = symbols("q1 q2 q3 q4 q5 q6")
    T03 = base_frame @ Rz_symb(q1) @ Tz(d[0]) @ Rx(1) @ Rz_symb(q2) @ Tx(d[1]) @ Rz_symb(q3) @ Tx(d[2]) @ Ry(-1)

    expr2 = T03[0, 3] - Pc[0, 3]
    expr3 = T03[2, 3] - Pc[2, 3]
    expr2 = expr2.subs(q1, q1_sol)

    q2_sol, q3_sol = nsolve((expr2, expr3), (q2, q3), (1, 1))
    q2_sol = float(q2_sol)
    q3_sol = float(q3_sol)

    T01 = base_frame @ Rz(q1_sol) @ Tz(d[0]) @ Rx(1)
    T12 = Rz(q2_sol) @ Tx(d[1])
    T23 = Rz(q3_sol) @ Tx(d[2]) @ Ry(-1)

    T03 = T01 @ T12 @ T23
    R36 = np.linalg.inv(T03) @ Pc @ T34(q, d) @ T45(q, d) @ T56(q, d)

    q4_sol = m.atan2(R36[1, 2], R36[0, 2])

    q5_sol = np.arccos(R36[2, 2])

    q6_sol = m.atan2(R36[2, 1], -R36[2, 0])

    return [q1_sol, q2_sol, q3_sol, q4_sol, q5_sol, q6_sol]


# Function to find Jacobian matrix
def Jacobian(frames, q, flag="ee"):
    # Create empty matrix 6xn
    J = np.zeros((6, len(q)))
    if flag == "full":
        for i in range(len(q)):
            rot_matr = frames[i][:3, 2]  # 3rd column of tranlation matrix
            dis_matr = (frames[-1] - frames[i])[:3, 3]  # last column of translation matrix
            top_el = np.cross(rot_matr.transpose(), dis_matr.transpose()).transpose()  # cross product
            # Fill elemnts of J_i element
            J[0, i] = top_el[0]
            J[1, i] = top_el[1]
            J[2, i] = top_el[2]
            J[3, i] = rot_matr[0]
            J[4, i] = rot_matr[1]
            J[5, i] = rot_matr[2]
    else:
        print("\nERROR: wrong input.\nCannot find Jacobian matrices since there is no frames.\n")
    return J


# Check singularity of Jacobian matrix
def checkJacSing(J):
    return np.linalg.det(J) == 0


# Return cartesian velocity
def cartesian_vel(J, q_dot):
    return J @ q_dot


# Calculate CoM of each link
def find_com(d, dc, q_symb):
    com1 = Rz_symb(q_symb[0]) @ Tz(dc[0]) @ Rx(1)

    com2 = Rz_symb(q_symb[0]) @ Tz(d[0]) @ Rx(1) \
           @ Rz_symb(q_symb[1]) @ Tx(dc[1])

    com3 = Rz_symb(q_symb[0]) @ Tz(d[0]) @ Rx(1) \
           @ Rz_symb(q_symb[1]) @ Tx(d[1]) \
           @ Rz_symb(q_symb[2]) @ Tx(dc[2]) @ Ry(-1)

    com4 = Rz_symb(q_symb[0]) @ Tz(d[0]) @ Rx(1) \
           @ Rz_symb(q_symb[1]) @ Tx(d[1]) \
           @ Rz_symb(q_symb[2]) @ Tx(d[2]) @ Ry(-1) \
           @ Rz_symb(q_symb[3]) @ Tz(dc[3]) @ Rx(-1)

    com5 = Rz_symb(q_symb[0]) @ Tz(d[0]) @ Rx(1) \
           @ Rz_symb(q_symb[1]) @ Tx(d[1]) \
           @ Rz_symb(q_symb[2]) @ Tx(d[2]) @ Ry(-1) \
           @ Rz_symb(q_symb[3]) @ Tz(d[3]) @ Rx(-1) \
           @ Rz_symb(q_symb[4]) @ Ty(dc[4]) @ Rx(1)

    com6 = Rz_symb(q_symb[0]) @ Tz(d[0]) @ Rx(1) \
           @ Rz_symb(q_symb[1]) @ Tx(d[1]) \
           @ Rz_symb(q_symb[2]) @ Tx(d[2]) @ Ry(-1) \
           @ Rz_symb(q_symb[3]) @ Tz(d[3]) @ Rx(-1) \
           @ Rz_symb(q_symb[4]) @ Ty(d[4]) @ Rx(1) \
           @ Rz_symb(q_symb[5]) @ Tz(dc[5])
    return np.array([com1, com2, com3, com4, com5, com6])


# Find J_v of each link
def find_Jv(com, q_symb):
    Jv = np.zeros((6, 3, 6), dtype=np.object_)
    for k in range(6):
        for i in range(3):
            for j in range(6):
                Jv[k, i, j] = diff(com[k, i, 3], q_symb[j])
    return Jv


# Find J_w of each link
def find_Jw(com):
    u = [np.array([0, 0, 1]), com[0, :3, 2], com[1, :3, 2], com[2, :3, 2], com[3, :3, 2], com[4, :3, 2]]
    Jw = np.zeros((6, 3, 6), dtype=np.object_)
    for n in range(6):
        for j in range(n + 1):
            Jw[n, :, j] = u[j]
    return Jw


# Find result inertia matrix M(q)
def find_inertia_matr(mass, com, I, Jv, Jw):
    Mq = np.zeros((6, 6))
    for i in range(6):
        Mq = Mq + mass[i] * np.transpose(Jv[i]) @ Jv[i] \
             + np.transpose(Jw[i]) @ com[i][:3, :3] @ I[i] \
             @ np.transpose(com[i][:3, :3]) @ Jw[i]
    return Mq


# Find result Coriolis matrix C(q, q_dot)
def find_Cor_matr(Mq, q_symb, q_dot):
    Cor = np.zeros((6, 6), dtype=np.object_)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                Cor[i, j] = Cor[i, j] + 1 / 2 * (
                            diff(Mq[i, j], q_symb[k]) + diff(Mq[i, k], q_symb[k]) - diff(Mq[j, k], q_symb[i])) * q_dot[
                                k]
    return Cor


# Find resul gravity matrix g(q)
def find_grav_matr(Jv, mass, g0):
    grav = np.zeros((6, 1), dtype=np.object_)
    for k in range(len(Jv)):
        for i in range(len(Jv)):
            grav[k] = grav[k] + mass[k] * np.transpose(Jv[k][:, i]) @ g0
    return grav


# Default base frame
base_frame = np.identity(4)
flag = "full"
# Values of angles
q = [np.pi / 3, np.pi / 3, np.pi / 2, np.pi / 2, np.pi / 6, np.pi / 6]
# Lengths of links
d = [1, 2, 3, 4, 5, 6]
# Find transformation matrices
frames = transform_base(q=q, trans=base_frame, flag=flag)

# Solve Inverse Kinematics
q_sol = IK_solve(frames)
print("Inverse kinematics solution:\n")
for i in range(len(q_sol)):
    print(f"q{i + 1} is {m.degrees(q[i])} and found as {m.degrees(q_sol[i])}")

# Find Jacobian matrix
J = Jacobian(frames, q, flag=flag)
print(f"\nJacobian matrix is singular: {checkJacSing(J)}")

# Calculate cartesian velocity
q1_dot, q2_dot, q3_dot, q4_dot, q5_dot, q6_dot = symbols("q1_dot q2_dot q3_dot q4_dot q5_dot q6_dot")
q_dot = [q1_dot, q2_dot, q3_dot, q4_dot, q5_dot, q6_dot]
print("\nCartesian velocity is found\n")




# Differential kinematics
print("\nMove to Differential kinematics\n")
q0, q1, q2, q3, q4, q5 = symbols("q0 q1 q2 q3 q4 q5")
q_symb = [q0, q1, q2, q3, q4, q5]
dc = [di / 2 for di in d]

# Find CoM
com = find_com(d, dc, q_symb)
print("Found CoM.")
symb_Jv = find_Jv(com, q_symb)
print("Found Jv.")
symb_Jw = find_Jw(com)
print("Found Jw.\n")

print("Calculating M(q)...")
# Masses of links
mass = [4, 5, 6, 7, 8, 9]

# Tensor of Inertia for cubic links
I = np.zeros((6, 3, 3))
for i in range(3):
    I[i][0, 0] = mass[i] / 12 * (2 * d[i] ** 2)
    I[i][1, 1] = mass[i] / 12 * (2 * d[i] ** 2)
    I[i][2, 2] = mass[i] / 12 * (2 * d[i] ** 2)

# Find result inertia matrix
Mq = find_inertia_matr(mass, com, I, symb_Jv, symb_Jw)
print("M(q) is found.\n")

# Find result Coriolis matrix
print("Calculating C(q, q_dot)...")
Cor_matr = find_Cor_matr(Mq, q_symb, q_dot)
print("C(q, q_dot) is found.\n")

# Find result gravity matrix
print("Calculating g(q)...")
g0 = np.array([[0], [0], [-9.8]])
grav_matr = find_grav_matr(symb_Jv, mass, g0)
print("g(q) is found.\n")

print("\nAll calculations complete.\n")
