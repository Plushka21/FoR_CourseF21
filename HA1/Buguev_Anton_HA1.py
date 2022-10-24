import numpy as np
import math as m
from sympy import *


# Generate rotation matrices around each axes on certain angle
def Rx(angle):
    return np.matrix([[1, 0, 0, 0],
                      [0, np.cos(angle), -np.sin(angle), 0],
                      [0, np.sin(angle), np.cos(angle), 0],
                      [0, 0, 0, 1]])


def Ry(angle):
    return np.matrix([[np.cos(angle), 0, np.sin(angle), 0],
                      [0, 1, 0, 0],
                      [-np.sin(angle), 0, np.cos(angle), 0],
                      [0, 0, 0, 1]])


def Rz(angle):
    return np.matrix([[np.cos(angle), -np.sin(angle), 0, 0],
                      [np.sin(angle), np.cos(angle), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


# Generate translation matrices along each axes on certain distance
def Tx(d):
    return np.matrix([[1, 0, 0, d],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


def Ty(d):
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, d],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


def Tz(d):
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, d],
                      [0, 0, 0, 1]])


# Solve forward kinematics
def FK_solve(q, d, trans, flag="ee"):
    trans_matrices = [trans,
                      Rz(q[0]) * Tz(d[0]) * Rx(np.pi / 2),
                      Rz(q[1]) * Tx(d[1]),
                      Rz(q[2]) * Tx(d[2]) * Ry(-np.pi / 2),
                      Rz(q[3]) * Rx(-np.pi / 2),
                      Rz(q[4]) * Rx(np.pi / 2),
                      Rz(q[5]) * Tz(q[5])]

    transformation = np.identity(4)
    steps = []
    for i in range(len(trans_matrices)):
        transformation = transformation * trans_matrices[i]
        steps.append(transformation)

    print(f"Transformation matrix from 0 to 6 is\n{steps[-1]}")
    # Return either all steps or only final matrix
    if flag == "ee":
        return steps[-1]
    elif flag == "full":
        return steps


def transform_base(trans, q=None, flag="ee", d=None):
    if d is None:
        d = [1, 2, 3, 4, 5, 6]
    if q is None:
        q = [1, 2, 3, 4, 5, 6]
    T = FK_solve(q, d, trans, flag)
    return T


# Generate symbolic rotation matrices around each axes
def Rx_symb(cos_symb, sin_symb):
    return np.matrix([[1, 0, 0, 0],
                      [0, cos_symb, -sin_symb, 0],
                      [0, sin_symb, cos_symb, 0],
                      [0, 0, 0, 1]])


def Ry_symb(cos_symb, sin_symb):
    return np.matrix([[cos_symb, 0, sin_symb, 0],
                      [0, 1, 0, 0],
                      [-sin_symb, 0, cos_symb, 0],
                      [0, 0, 0, 1]])


def Rz_symb(cos_symb, sin_symb):
    return np.matrix([[cos_symb, -sin_symb, 0, 0],
                      [sin_symb, cos_symb, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


# Solve inverse kinematics
def IK_solve(ee_frame, base_frame=np.identity(4), d=None, flag="ee"):
    if d is None:
        d = [1, 2, 3, 4, 5, 6]
    if flag == "ee":
        Pc = ee_frame * np.linalg.inv(Tz(d[5]))
    elif flag == "full":
        Pc = ee_frame[-1] * np.linalg.inv(Tz(d[5]))

    xc, yc, zc = Pc[0, 3], Pc[1, 3], Pc[2, 3]
    print(f"xc is {xc}")
    print(f"yc is {yc}")
    print(f"zc is {zc}")

    q1 = m.atan2(yc, xc)
    print(f"q1 is {m.degrees(q1)}")

    c3 = (yc**2 + xc**2 - (d[1]**2 + d[2]**2)) / (2 * d[1] * d[2])
    s3 = np.sqrt(1-c3**2)
    q3 = m.atan2(s3, c3)
    print(f"q3 is {m.degrees(q3)}")

    s2 = (yc * (d[1] + d[2] * c3) - xc * d[2] * s3) / (d[1] ** 2 + d[2] ** 2 + 2 * d[1] * d[2] * c3)
    c2 = (xc * (d[1] + d[2] * c3) + yc * d[2] * s3) / (d[1] ** 2 + d[2] ** 2 + 2 * d[1] * d[2] * c3)
    q2 = m.atan2(s2, c2)
    print(f"q2 is {m.degrees(q2)}")

    T03 = np.identity(4) * Rz(q[0]) * Tz(d[0]) * Rx(np.pi / 2) * Rz(q[1]) * Tx(d[1]) * Rz(q[2]) * Tx(d[2]) * Ry(-np.pi / 2)
    R03 = T03[:3, :3]
    R36 = np.linalg.inv(R03) * Pc[:3, :3]

    q4 = m.atan2(R36[1, 2], R36[0, 1])
    print(f"q4 is {m.degrees(q4)}")

    q5 = np.arccos(R36[2, 2])
    print(f"q5 is {m.degrees(q5)}")

    q6 = m.atan2(-R36[2, 1], R36[2, 0])
    print(f"q6 is {m.degrees(q6)}")

    return 0


base_frame = np.identity(4)
flag = "full"
q = [np.pi / 3, np.pi/3, np.pi/2, np.pi/2, np.pi/6, np.pi/6]
T = transform_base(q=q, trans=base_frame, flag=flag)

IK_solve(T, flag=flag)

'''c1, c2, c3, c4, c5, c6, s1, s2, s3, s4, s5, s6, d1, d2, d3, d6 = \
    symbols("c1 c2 c3 c4 c5 c6 s1 s2 s3 s4 s5 s6 d1 d2 d3 d6")

T03 = Rz_symb(c1, s1) * Tz(d1) * Rx(np.pi/2) * Rz_symb(c2, s2) * Tx(d2) * Rz_symb(c3, s3) * Tx(d3) * Ry(-np.pi/2)
#print(T03)'''
