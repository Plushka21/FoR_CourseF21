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
            rot_matr = frames[i][:3, 2] # 3rd column of tranlation matrix
            dis_matr = (frames[-1] - frames[i])[:3, 3] # last column of translation matrix
            top_el = np.cross(rot_matr.transpose(), dis_matr.transpose()).transpose() # cross product
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


base_frame = np.identity(4)
flag = "full"
q = [np.pi / 3, np.pi / 3, np.pi / 2, np.pi / 2, np.pi / 6, np.pi / 6]
d = [1, 2, 3, 4, 5, 6]
frames = transform_base(q=q, trans=base_frame, flag=flag)

# TODO: Find all possible solutions
q_sol = IK_solve(frames)

for i in range(len(q_sol)):
    print(f"q{i + 1} is {m.degrees(q[i])} and found as {m.degrees(q_sol[i])}")

J = Jacobian(frames, q, flag=flag)
print(f"\nJacobian matrix is singular: {checkJacSing(J)}")

q_dot = np.array([np.pi / 10, np.pi / 10, np.pi / 10, np.pi / 10, np.pi / 10, np.pi / 10])
print(f"\nCartesian velocity :{cartesian_vel(J, q_dot)}")
