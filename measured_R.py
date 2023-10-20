import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

"L'objectif de ces fonctions est de déterminer une estimation de R,"
"sur la base des seules mesures fournies par l'accéléromètre et le magnétomètre."
"................................................................................"


def Estimate_G(Gamma):
    "On calcule une bonne estimation de la pesanteur à partir des mesures de l'accéléromètre"
    N = np.shape(Gamma)[0]  # le nombre d'échantillons

    Norm_G = 0
    for measurement in Gamma:
        Norm_G = Norm_G + np.linalg.norm(measurement)

    return np.array([[0, 0, Norm_G/N]])


def Estimate_M(Mag, I):
    "Donne une bonne estimation du champ magnétique à partir des mesures du magnétomètre"
    N = np.shape(Mag)[0]  # le nombre d'échantillons
    Norm_B = 0
    for measurement in Mag:
        Norm_B = Norm_B + np.linalg.norm(measurement)

    B = Norm_B/N
    # L'angle d'inclinaison magnétique, qui permet de déterminer la composante verticale et horizontale du champ magnétique
    I = (I*np.pi)/180

    return np.array([[B*np.cos(I), 0, B*np.sin(I)]])


def Estimate_R_r7_r8_r9(estimate_g, local_acceleration):
    "Donne la troisième ligne de la matrice R grâce à la valeur estimée de g0 et la valeur mesurée par l'accéléromètre"

    r7_r8_r9 = -local_acceleration/estimate_g[0][2]  # third row of R

    return r7_r8_r9


def roll_pitch_yaw(R):
    "Retourne les angles de roulis (roll), de tangage (pitch) et de lacet (yaw) de la matrice de rotation R"

    # deux degrés de liberté sont résolus par l'accéléromètre
    roll = (np.arctan2(R[2][1], R[2][2]))*180/np.pi  # 3rd ligne (r7_r8_r9)
    pitch = np.arctan2(-R[2][0], np.sqrt(R[2][1]**2+R[2][2]**2)
                       )*(180/np.pi)  # 3rd ligne (r7_r8_r9)
    yaw = np.arctan2(R[1][0], R[0][0])*(180/np.pi)

    return [roll, pitch, yaw]


def Estimate_R_r1_r2_r3(r7_r8_r9, estimate_m, local_magnetic_field):
    "Donne ce que devrait être la première ligne de la matrice R grâce à la valeur estimée de B0 et en respectant les mesures données par le magnétomètre"

    r1_r2_r3 = (local_magnetic_field -
                estimate_m[0][2]*r7_r8_r9)/estimate_m[0][0]

    return r1_r2_r3


def objectif(r1_r2_r3, estimate_m, local_magnetic_field, r7_r8_r9):
    "Donne la valeur de |B-R'*B0|, à minimiser par la suite pour que R reste dans SO(3)"

    objective = estimate_m[0][0]*r1_r2_r3 + \
        estimate_m[0][2]*r7_r8_r9-local_magnetic_field

    return np.linalg.norm(objective)**2  # +(np.linalg.norm(r1_r2_r3)**2-1)**2


def r1_r2_r3_by_minimization(r7_r8_r9, r1_r2_r3_0, estimate_m, mag_i):
    "Donne la première ligne de la matrice R en minimisant la fonction objectif et en respectant la contrainte R dans SO(3)"

    x0 = r1_r2_r3_0  # ce que devrait être la première ligne de la matrice R
    lbnd = [0]
    upbnd = [0]
    lin_cons = LinearConstraint(r7_r8_r9, lbnd, upbnd)

    r1_r2_r3 = minimize(objectif, x0, args=(
        estimate_m, mag_i, r7_r8_r9), constraints=lin_cons)['x']

    return r1_r2_r3/np.linalg.norm(r1_r2_r3)  # first row of R


def estimate_r4_r5_r6(r1_r2_r3, r7_r8_r9):
    "Donne la deuxième ligne de la matrice R sachant que les lignes de R forment une base orthonormée"

    return np.cross(r7_r8_r9, r1_r2_r3)  # the second row of R


def estimated_R(r1_r2_r3, r4_r5_r6, r7_r8_r9):
    "Donne la matrice R de SO(3)"

    return np.concatenate((r1_r2_r3.reshape((1, 3)), r4_r5_r6.reshape((1, 3)), r7_r8_r9.reshape((1, 3))), axis=0)
