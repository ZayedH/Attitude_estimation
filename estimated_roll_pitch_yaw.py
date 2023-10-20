from scipy.io import loadmat
from measured_R import *
from to_filter_R import *


# Extraction des données du fichier matlab fourni
data = loadmat("DonnesBrutesCapteurs_Sysnav2014.mat")
Gamma = data['Gamma']
Mag = data['Mag']*10**(-4)  # on peut garder en Gauss, ça n'affecte pas
Time = data['Time']
Omega = data['Omega']
np.save('Others/Omega.npy', Omega)
# _______ Une estimation de l'angle d'inclinaison magnétique
I = 64.46  # obtenu à partir du World Magnetic Model (WMM2010)

#___Une estimation des grandeurs physiques à mesurer____#
estimate_g = Estimate_G(Gamma)
estimate_m = Estimate_M(Mag, I)
# _______
Roll_measered = []
Pitch_measered = []
Yaw_measered = []
# ________
Roll_filtered = []
Pitch_filtered = []
Yaw_filtered = []
gyro_bias = []
#___Les paramètres du filtre___#
kI = 0.3  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<---¬
kP = 1                                                         # ^
# ______________#                                               # ^
b_filter = np.array([[0, 0, 0]])  # pour que le filtre converge # ^
T = 0.0020  # le pas de temps
N = np.shape(Gamma)[0]
for i in range(N):
    r7_r8_r9 = Estimate_R_r7_r8_r9(estimate_g, Gamma[i])

    r1_r2_r3_0 = Estimate_R_r1_r2_r3(r7_r8_r9, estimate_m, Mag[i])

    x = r1_r2_r3_by_minimization(r7_r8_r9, r1_r2_r3_0, estimate_m, Mag[i])

    r4_r5_r6 = estimate_r4_r5_r6(x, r7_r8_r9)

    R = estimated_R(x, r4_r5_r6, r7_r8_r9)
    if(i == 0):
        R_filter = R  # pour converger

    measured_roll_pitch_yaw = roll_pitch_yaw(R)

    Roll_measered.append(measured_roll_pitch_yaw[0])
    Pitch_measered.append(measured_roll_pitch_yaw[1])
    Yaw_measered.append(measured_roll_pitch_yaw[2])

    filtered_roll_pitch_yaw = roll_pitch_yaw(R_filter)

    Roll_filtered.append(filtered_roll_pitch_yaw[0])
    Pitch_filtered.append(filtered_roll_pitch_yaw[1])
    Yaw_filtered.append(filtered_roll_pitch_yaw[2])

    gyro_bias.append(b_filter)

    R_tild = R_filter.T@R

    vex_pi_aR_tild = vex(Pi_a(R_tild))
    omega_i_x = inv_vex(Omega[i]-b_filter+kP*vex_pi_aR_tild)
    A_i = filter_A_k(omega_i_x, T)
    R_filter = R_filter@A_i
    b_filter = b_filter-T*kI*vex_pi_aR_tild


# __Saving roll, pitch and yaw measurements (obtained by relying exclusively on measurements)
np.save('Measured_Roll_Pitch_Yow/measured_Roll.npy', Roll_measered)
np.save('Measured_Roll_Pitch_Yow/measured_Pitch.npy', Pitch_measered)
np.save('Measured_Roll_Pitch_Yow/measured_Yaw.npy', Yaw_measered)
# _________
np.save('Measured_Roll_Pitch_Yow/time.npy', Time)
# __Filtered roll, pitch and yaw
np.save('Measured_Roll_Pitch_Yow/filtered_Roll.npy', Roll_filtered)
np.save('Measured_Roll_Pitch_Yow/filtered_Pitch.npy', Pitch_filtered)
np.save('Measured_Roll_Pitch_Yow/filtered_Yaw.npy', Yaw_filtered)
# _Estimated gyroscope bias
np.save('Measured_Roll_Pitch_Yow/gyro_bias.npy', gyro_bias)
