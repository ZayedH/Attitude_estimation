import numpy as np

"Ce fichier permet de simuler les angles de roulis, de tangage et de lacet"
"Il permet également de simuler les mesures de l'accéléromètre, du magnétomètre et du gyroscope"
"......................................................................................................."


def simulated_R(roll, pitch, yaw):
    "Retourne la matrice de rotation"
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    R_x = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    return R_z@R_y@R_x


def simulated_acceleration(simulated_R, estimate_g):
    "Retourne l'accélération locale mesurée dans le body frame"

    return -estimate_g@simulated_R


def simulated_magnetic_vector(simulated_R, estimate_m):
    "Retourne le champ magnétique local mesuré dans le body frame"

    return estimate_m@simulated_R


def simulated_angular_velocity(simulated_R_t1, simulated_R_t2, Dt2_t1):
    "Retourne la vitesse angulaire"

    omega_x = simulated_R_t1.T@(simulated_R_t2-simulated_R_t1)/Dt2_t1
    good_omega_x = (omega_x-omega_x.T)/2

    return good_omega_x
