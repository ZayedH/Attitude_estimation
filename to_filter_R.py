import numpy as np

"Contient des fonctions servant le filtrage de R"
"..............................................."


def vex(M):
    " Retourne m, telque vex(M)=m"

    return np.array([[M[2][1], M[0][2], M[1][0]]])


def inv_vex(m):
    "Retourne M, telque vex(M)=m"
    m1 = m[0][0]
    m2 = m[0][1]
    m3 = m[0][2]

    return np.array([[0, -m3, m2],
                     [m3, 0, -m1],
                     [-m2, m1, 0]])


def Pi_a(H):

    return 0.5*(H-H.T)


def filter_A_k(omega_k, T):
    "Retourne exp(Tomega_k) en utilisant la formule de Rodrigues"
    norm_omega_k = np.linalg.norm(omega_k)  # The Frobenius norm
    normed_omega_k = omega_k/norm_omega_k
    theta = T*norm_omega_k
    A_k = np.identity(3)+normed_omega_k*np.sin(theta) + \
        (normed_omega_k@normed_omega_k)*(1-np.cos(theta))
    return A_k
