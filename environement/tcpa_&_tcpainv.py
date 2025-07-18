# Fonction python pour les calcul des tcpa/cpa et du point d'interception

import math
import numpy as np

def calculer_TCPA(x_a, y_a, cap_a, vit_a, x_b, y_b, cap_b, vit_b):
    """
    Calcule le Time to Closest Point of Approach (TCPA)

    Paramètres :
        x_a, y_a : float — Coordonnées du navire A.
        cap_a : float — Cap (en degrés) du navire A.
        vit_a : float — Vitesse du navire A.
        x_b, y_b : float — Coordonnées du navire B.
        cap_b : float — Cap (en degrés) du navire B.
        vit_b : float — Vitesse du navire B.

    Retour :
        TCPA : float — Temps en secondes avant d'atteindre le CPA.
    """

    cap_a_d = np.radians(cap_a)
    cap_b_d = np.radians(cap_b)
    
    vx_a, vy_a = np.cos(cap_a_d) * vit_a, np.sin(cap_a_d) * vit_a
    vx_b, vy_b = np.cos(cap_b_d) * vit_b, np.sin(cap_b_d) * vit_b
    
    # Vitesse relative
    vr_x = vx_b - vx_a
    vr_y = vy_b - vy_a

    # Position relative
    pr_x = x_b - x_a
    pr_y = y_b - y_a

    # Produit scalaire P_R . V_R
    vr_norme_carre = vr_x**2 + vr_y**2

    if vr_norme_carre == 0:
        return 0  # Les bateaux se déplacent dans la même direction à la même vitesse : le CPA est constant


    TCPA = -(pr_x * vr_x + pr_y * vr_y) / vr_norme_carre
    return max(TCPA, 0)  # On ne considère que le futur (TCPA >= 0)

def calculer_TCAP_et_CPA(x_a, y_a, cap_a, vit_a, x_b, y_b, cap_b, vit_b):
    """
    Calcule le Time to Closest Point of Approach (TCPA) et la distance au CPA.

    Paramètres :
        x_a, y_a : float — Coordonnées du navire A.
        cap_a : float — Cap (en degrés) du navire A.
        vit_a : float — Vitesse du navire A.
        x_b, y_b : float — Coordonnées du navire B.
        cap_b : float — Cap (en degrés) du navire B.
        vit_b : float — Vitesse du navire B.

    Retour :
        TCPA : float — Temps en secondes avant d'atteindre le CPA.
        CPA : float — Distance minimale atteinte entre les deux navires.
    """


    cap_a_d = np.radians(cap_a)
    cap_b_d = np.radians(cap_b)
    
    vx_a, vy_a = np.cos(cap_a_d) * vit_a, np.sin(cap_a_d) * vit_a
    vx_b, vy_b = np.cos(cap_b_d) * vit_b, np.sin(cap_b_d) * vit_b
    
    # Vitesse relative
    vr_x = vx_b - vx_a
    vr_y = vy_b - vy_a

    # Position relative
    pr_x = x_b - x_a
    pr_y = y_b - y_a

    # Produit scalaire P_R . V_R
    vr_norme_carre = vr_x**2 + vr_y**2

    if vr_norme_carre == 0:
        TCPA = 0  
    else :    
        TCPA = -(pr_x * vr_x + pr_y * vr_y) / vr_norme_carre
        TCPA = max(TCPA, 0)

    # Position relative au moment du CPA
    cpa_x = pr_x + TCPA * vr_x
    cpa_y = pr_y + TCPA * vr_y

    # Distance au CPA
    CPA = math.sqrt(cpa_x**2 + cpa_y**2)
    return TCPA, CPA

def Point_d_interception(x_a, y_a, vit_a, x_b, y_b, cap_b, vit_b):
    """
    Calcule le point d'interception que le navire A doit viser pour intercepter le navire B.

    Paramètres :
        x_a, y_a : float — Coordonnées du navire A.
        vit_a : float — Vitesse du navire A.
        x_b, y_b : float — Coordonnées du navire B.
        vit_b : float — Vitesse du navire B.
        cap_b : float — Cap (en degrés) du navire B.

    Hypothèses :
        - Les deux navires se déplacent en ligne droite à vitesse constante.
        - On résout une équation quadratique pour déterminer le temps t au bout duquel le navire A peut intercepter B.
        - Si aucune interception n’est possible, on retourne la position future de B après un temps arbitraire t.

    Retour :
        target_vec : np.array — Coordonnées [x, y] du point d’interception.
    """

    theta_b = np.radians(cap_b)

    dx = x_b - x_a
    dy = y_b - y_a
    d = np.array([dx, dy])
    v = vit_b * np.array([np.cos(theta_b), np.sin(theta_b)])

    # Résolution de l'équation quadratique : A·t² + B·t + C = 0
    A = np.dot(v, v) - vit_a**2
    B = 2 * np.dot(d, v)
    C = np.dot(d, d)

    interception_possible = True
    if abs(A) < 1e-8:
        if abs(B) < 1e-8:
            interception_possible = False
        else:
            t = -C / B
            if t <= 0:
                interception_possible = False
    else:
        delta = B**2 - 4*A*C
        
        if delta >= 0:
            t1 = (-B - np.sqrt(delta)) / (2*A)
            t2 = (-B + np.sqrt(delta)) / (2*A)
            t_candidates = [t for t in (t1, t2) if t > 0]
            if not t_candidates:
                interception_possible = False
            else :
                t = min(t_candidates)

        else : 
            interception_possible = False

    if interception_possible == False:
        t = 10.0  # Valeur arbitraire
        print("Pas d'interception : je vise la position future à t =", t)

    # Calcul du point d'interception visé
    xt = x_b + v[0] * t
    yt = y_b + v[1] * t

    target_vec = np.array([round(xt,2), round(yt,2)]) # Vecteur du chasseur vers le point d’interception

    return target_vec


#------------------------------------------------Zone de test------------------------------------------------#

def test_calculer_TCAP_et_CPA():
    tcpa, cpa = calculer_TCAP_et_CPA(0, 0, 0, 10, 250, 250, 270, 10)
    print("Test 1")
    print("TCPA attendu ≈ 25.0, obtenu :", round(tcpa, 2))
    print("CPA attendu ≈ 0.0, obtenu :", round(cpa, 2))
    print("")

    tcpa, cpa = calculer_TCAP_et_CPA(250, 250, 0, 10, 750, 500, 270, 10)
    print("Test 2")
    print("TCPA attendu ≈ 37.5, obtenu :", round(tcpa, 2))
    print("CPA attendu ≈ 176.78, obtenu :", round(cpa, 2))
    print("")

    tcpa, cpa = calculer_TCAP_et_CPA(250, 750, 315, 10, 100, 750, 225, 10)
    print("Test 3")
    print("TCPA attendu ≈ 150, obtenu :", round(tcpa, 2))
    print("CPA attendu ≈ 0, obtenu :", round(cpa, 2))
    print("")

    tcpa, cpa = calculer_TCAP_et_CPA(250, 250, 0, 10, 500, 500, 0, 10)
    print("Test 4")
    print("TCPA attendu ≈ 0, obtenu :", round(tcpa, 2))
    print("CPA attendu ≈ 353.55, obtenu :", round(cpa, 2))



def test_Point_d_interception():
    point = Point_d_interception(750, 500, 10, 250, 250, 0, 10)
    print("Test 1")
    print("Point d'interception attendu (562.5, 250), obtenu :", point)
    print("")

    # Situation de rattrapage impossible
    print("Test 2")
    point = Point_d_interception(250, 500, 10, 500, 500, 0, 10)
    print("Point d'interception attendu (600, 500), obtenu :", point)
    print("")

#------------------------------------------------Main------------------------------------------------#

if __name__ == "__main__":
    print("\n=== Test calculer_TCAP_et_CPA ===")
    test_calculer_TCAP_et_CPA()
    print("\n=== Test Point_d_interception ===")
    test_Point_d_interception()
