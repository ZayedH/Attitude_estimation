import numpy as np
from to_filter_R import *
from simulated_R import *
from measured_R import *


"Ce fichier est utilisé pour tester les performances du filtre sur des mesures simulées avec et sans bruit"
"........................................................................................................."

# __Generation of roll, pitch and yaw
N = 20000
time = np.linspace(0, 40, N)
f0 = 0.2
roll0 = 20
pitch0 = 10
yaw0 = 5
roll = roll0*np.cos(time*np.pi*f0)
pitch = pitch0*np.cos(time*np.pi*f0)
yaw = yaw0*np.cos(time*np.pi*f0)
dt = time[1]-time[0]

# __Saving real roll, pitch and yaw angles in order to be compared with  the filtered values
np.save('Simulated_Roll_Pitch_Yaw/simulated_Roll.npy', roll)
np.save('Simulated_Roll_Pitch_Yaw/simulated_Pitch.npy', pitch)
np.save('Simulated_Roll_Pitch_Yaw/simulated_Yaw.npy', yaw)
np.save('Simulated_Roll_Pitch_Yaw/time.npy', time)

#___Les grandeurs physiques à mesurer___#
g0 = 10
# pour la simulation ces valeurs n'ont pas d'importance on peut mettre ce que l'on veut
Bh = 2*10**(-5)
Bv = 4*10**(-5)
estimate_g = np.array([[0, 0, g0]])  # la pesanteur
estimate_m = np.array([[Bh, 0, Bv]])  # le champ magnétique


#___Générations des grandeurs physiques (mesurées) dans le référentiel local___#
Omega = []
Gamma = []
Mag = []

for i in range(N):
    if(i < N-1):
        Rt1 = simulated_R(roll[i]*np.pi/180, pitch[i]
                          * np.pi/180, yaw[i]*np.pi/180)
        Rt2 = simulated_R(roll[i+1]*np.pi/180, pitch[i+1]
                          * np.pi/180, yaw[i+1]*np.pi/180)
        omega_i = simulated_angular_velocity(Rt1, Rt2, dt)
        Omega.append(vex(omega_i)+0.001*np.cos(0.01*N*dt) +
                     0.001*np.random.randn(3))
    else:
        roll_i_1 = roll0*np.cos(N*dt*np.pi*f0)
        pitch_i_1 = pitch0*np.cos(N*dt*np.pi*f0)
        yaw_i_i = yaw0*np.cos(N*dt*np.pi*f0)
        Rt1 = simulated_R(roll_i_1*np.pi/180, pitch_i_1 *
                          np.pi/180, yaw_i_i*np.pi/180)
        omega_i = simulated_angular_velocity(Rt2, Rt1, dt)
        Omega.append(vex(omega_i)+0.001*np.cos(0.01*N*dt) +
                     0.001*np.random.randn(3))

for i in range(N):
    R = simulated_R(roll[i]*np.pi/180, pitch[i]*np.pi/180, yaw[i]*np.pi/180)
    # Processus de Bernoulli (pour dire que parfois nous avons une bonne mesure et parfois non)
    gamma_i = simulated_acceleration(
        R, estimate_g+(0.1*np.random.randn(3))*np.random.binomial(1, 0.1, 3).reshape((1, 3)))
    # (dans la majorité des cas, on a des mauvaises mesures par rapport à l'accéléromètre)
    mag_i = simulated_magnetic_vector(
        R, estimate_m+(0.00001*np.random.randn(3))*np.random.binomial(1, 0.5, 3).reshape((1, 3)))
    Gamma.append(gamma_i)
    Mag.append(mag_i)

# _________________
Roll_measered = []
Pitch_measered = []
Yaw_measered = []
# __________________
Roll_filtered = []
Pitch_filtered = []
Yaw_filtered = []
gyro_bias = []
#___Les paramètres du filtre___#
kI = 0.3
kP = 1
# ______________
b_filter = np.array([[0, 0, 0]])  # pour que le filtre converge

for i in range(N):
    r7_r8_r9 = Estimate_R_r7_r8_r9(estimate_g, Gamma[i])

    r1_r2_r3_0 = Estimate_R_r1_r2_r3(r7_r8_r9, estimate_m, Mag[i])

    x = r1_r2_r3_by_minimization(r7_r8_r9, r1_r2_r3_0.T, estimate_m, Mag[i])

    r4_r5_r6 = estimate_r4_r5_r6(x, r7_r8_r9)

    R = estimated_R(x, r4_r5_r6, r7_r8_r9)

    measured_roll_pitch_yaw = roll_pitch_yaw(R)

    Roll_measered.append(measured_roll_pitch_yaw[0])
    Pitch_measered.append(measured_roll_pitch_yaw[1])
    Yaw_measered.append(measured_roll_pitch_yaw[2])

    if(i == 0):
        R_filter = R  # pour que le filtre converge
    R_tild = R_filter.T@R

    gyro_bias.append(b_filter)

    vex_pi_aR_tild = vex(Pi_a(R_tild))
    omega_i = inv_vex(Omega[i]-b_filter+kP*vex_pi_aR_tild)
    A_i = filter_A_k(omega_i, dt)
    R_filter = R_filter@A_i
    b_filter = b_filter-dt*kI*vex_pi_aR_tild

    filtered_roll_pitch_yaw = roll_pitch_yaw(R_filter)

    Roll_filtered.append(filtered_roll_pitch_yaw[0])
    Pitch_filtered.append(filtered_roll_pitch_yaw[1])
    Yaw_filtered.append(filtered_roll_pitch_yaw[2])


# __Saving roll, pitch and yaw measurements (obtained by relying exclusively on measurements)
np.save('Simulated_Roll_Pitch_Yaw/measured_Roll.npy', Roll_measered)
np.save('Simulated_Roll_Pitch_Yaw/measured_Pitch.npy', Pitch_measered)
np.save('Simulated_Roll_Pitch_Yaw/measured_Yaw.npy', Yaw_measered)

# __Filtered roll, pitch and yaw
np.save('Simulated_Roll_Pitch_Yaw/filtered_Roll.npy', Roll_filtered)
np.save('Simulated_Roll_Pitch_Yaw/filtered_Pitch.npy', Pitch_filtered)
np.save('Simulated_Roll_Pitch_Yaw/filtered_Yaw.npy', Yaw_filtered)
# __
np.save('Simulated_Roll_Pitch_Yaw/gyro_bias.npy', gyro_bias)
