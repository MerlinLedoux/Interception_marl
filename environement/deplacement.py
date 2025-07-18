import numpy as np
import math

def move(x, y, cap, vit, ax, ay, max_speed):
    cap = np.radians(cap)

    ax = ax if vit + ax >= 0.1 else vit - 0.1
    
    x3 = x + np.cos(cap) * (vit + ax) - np.sin(cap) * ay
    y3 = y + np.sin(cap) * (vit + ax) + np.cos(cap) * ay

    new_cap = math.atan2(y3 - y, x3 - x)
    vitesse = np.linalg.norm(np.array([x3 - x, y3 - y]))
    
    new_vit = vitesse if vitesse <= max_speed else max_speed

    new_x = x + np.cos(new_cap) * new_vit
    new_y = y + np.sin(new_cap) * new_vit
    new_cap = (new_cap + 2*np.pi) % (2*np.pi)
    
    return(new_x, new_y, np.degrees(new_cap), new_vit)


# def chasseur_simple(xc, yc, capc, vitc, xe, ye, cape, vite):
def chasseur_simple(xc, yc, capc, xe, ye):
    dx = xe - xc
    dy = ye - yc

    cap_rad = np.radians(capc)

    # Base du repère local (orienté selon le cap)
    # Avant = direction du cap (x local), Droite = perpendiculaire droite (y local)
    forward = np.array([np.cos(cap_rad), np.sin(cap_rad)])
    right = np.array([-np.sin(cap_rad), np.cos(cap_rad)])

    # Projeter le vecteur global vers l’éviteur dans le repère local du chasseur
    target_vec = np.array([dx, dy])
    ax = np.clip(np.dot(target_vec, forward), -1.0, 1.0)  # composante longitudinale
    ay = np.clip(np.dot(target_vec, right), -1.0, 1.0)    # composante latérale

    return(np.array([ax, ay]))

def angle_entre_cap_et_cible(xc, yc, xe, ye, cape):
    vecteur = [xe - xc, ye - yc]
    angle_vecteur = np.arctan2(vecteur[1], vecteur[0])
    
    delta = angle_vecteur - cape
    # Normalisation entre -pi et pi
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    
    return delta

def chasseur_moyen(xc, yc, capc, xe, ye, cape):
    capc_rad = np.radians(capc)
    cape_rad = np.radians(cape)

    dx = xe - xc
    dy = ye - yc
    dist = np.sqrt(dx**2 + dy**2)

    teta = angle_entre_cap_et_cible(xe, ye, xc, yc, cape_rad)
    dist_int = (dist / 2) / max(np.cos(teta), 0.1)

    xt = xe + dist_int * np.cos(cape_rad)
    yt = ye + dist_int * np.sin(cape_rad)

    target_vec = np.array([xt - xc, yt - yc])
    # print(target_vec)

    forward = np.array([np.cos(capc_rad), np.sin(capc_rad)])
    right = np.array([-np.sin(capc_rad), np.cos(capc_rad)])

    ax = np.clip(np.dot(target_vec, forward), -1.0, 1.0)
    ay = np.clip(np.dot(target_vec, right), -1.0, 1.0)  
    
    return np.array([ax, ay])

def chasseur_moyen_2(xc, yc, capc, vitc, xe, ye, cape, vite):
    capc_rad = np.radians(capc)
    cape_rad = np.radians(cape)

    dx = xe - xc
    dy = ye - yc
    dist = np.sqrt(dx**2 + dy**2)

    teta = angle_entre_cap_et_cible(xe, ye, xc, yc, cape_rad)
    teta = np.abs(teta)
    dist_int = (dist * vitc / (vitc + vite)) / max(np.cos(teta), 1e-2)

    xt = xe + dist_int * np.cos(cape_rad)
    yt = ye + dist_int * np.sin(cape_rad)

    target_vec = np.array([xt - xc, yt - yc])
    # print(target_vec)

    forward = np.array([np.cos(capc_rad), np.sin(capc_rad)])
    right = np.array([-np.sin(capc_rad), np.cos(capc_rad)])

    ax = np.clip(np.dot(target_vec, forward), -1.0, 1.0)
    ay = np.clip(np.dot(target_vec, right), -1.0, 1.0)  
    
    return np.array([ax, ay])

def chasseur_moyen_3(xc, yc, capc, vitc, xe, ye, cape, vite):
    capc_rad = np.radians(capc)
    cape_rad = np.radians(cape)

    dx = xe - xc
    dy = ye - yc
    dist = np.sqrt(dx**2 + dy**2)

    time_est = min(dist / max((vitc + 1e-3), 0.1), 10)

    xt = xe + vite * time_est * np.cos(cape_rad)
    yt = ye + vite * time_est * np.sin(cape_rad)

    target_vec = np.array([xt - xc, yt - yc])
    # print(target_vec)

    forward = np.array([np.cos(capc_rad), np.sin(capc_rad)])
    right = np.array([-np.sin(capc_rad), np.cos(capc_rad)])

    ax = np.clip(np.dot(target_vec, forward), -1.0, 1.0)
    ay = np.clip(np.dot(target_vec, right), -1.0, 1.0)  
    
    return np.array([ax, ay])

def chasseur_hard(xc, yc, capc, vitc, xe, ye, cape, vite):
    """
    Calcule le vecteur d'accélération que le chasseur doit adopter pour intercepter l'éviteur 
    en mouvement, en supposant que tous deux se déplacent à vitesse constante.

    Paramètres :
    xc, yc : float : Coordonnées (x, y) du chasseur.
    capc : float : Cap (orientation en degrés) du chasseur.
    vitc : float : Vitesse du chasseur.
    xe, ye : float : Coordonnées (x, y) de l'éviteur.
    cape : float : Cap (orientation en degrés) de l'éviteur.
    vite : float : Vitesse de l'éviteur.

    Description :
    La fonction comporte deux phases :

    1. Calcul du point d'interception :
        - On suppose que les deux agents se déplacent en ligne droite à vitesse constante.
        - On résout une équation quadratique pour déterminer le moment t où le chasseur 
          peut atteindre l'éviteur s'ils maintiennent leur direction et vitesse.
        - Le point d'interception (xt, yt) est calculé à partir de ce temps t.

    2. Calcul de l'accélération :
        - Le vecteur entre le chasseur et le point d’interception est projeté dans 
          le repère local du chasseur (défini par son cap).
        - Les composantes ax et ay représentent alors les directions vers lesquelles 
          le chasseur doit accélérer (avant et latéral) pour se diriger vers ce point.
    """
    # print(xc, yc, capc, vitc, xe, ye, cape, vite)
    theta_e = np.radians(cape)
    theta_c = np.radians(capc)
    
    dx = xe - xc
    dy = ye - yc
    d = np.array([dx, dy])
    v = vite * np.array([np.cos(theta_e), np.sin(theta_e)])

    # Résolution de l'équation quadratique : A·t² + B·t + C = 0
    A = np.dot(v, v) - vitc**2
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
            print(t1, t2)
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
    xt = xe + v[0] * t
    yt = ye + v[1] * t 
    target_vec = np.array([round(xt-xc,2), round(yt-yc,2)]) # Vecteur du chasseur vers le point d’interception
    print(xt, yt)
    print("tets")

    # Repère local du chasseur
    forward = np.array([np.cos(theta_c), np.sin(theta_c)])
    right = np.array([-np.sin(theta_c), np.cos(theta_c)])

    # Projection du vecteur vers la cible dans le repère du chasseur
    ax = np.clip(np.dot(target_vec, forward), -1.0, 1.0)
    ay = np.clip(np.dot(target_vec, right), -1.0, 1.0)  
    
    return np.array([ax, ay])


#----------------Zone de test----------------#


# print("Test 1 : Même direction, éviteur droit devant")
# res1 = chasseur_hard(
#     xc=0, yc=0, capc=0, vitc=1.0,   # chasseur au centre, cap 0°
#     xe=10, ye=0, cape=180, vite=1.0 # éviteur droit devant à 10m, vient vers le chasseur
# )
# print("Résultat :", res1)

# # Cas 2 : éviteur qui fuit en diagonale, chasseur orienté vers le nord
# print("\nTest 2 : Éviteur en fuite, diagonale")
# res2 = chasseur_hard(
#     xc=0, yc=0, capc=90, vitc=1.5,     # chasseur cap vers le nord
#     xe=5, ye=5, cape=45, vite=1.0      # éviteur s'éloigne en diagonale (nord-est)
# )
# print("Résultat :", res2)

# # Cas 3 :
# print("\nTest 3 : ")
# res3 = chasseur_hard(
#     xc=10, yc=0, capc=0, vitc=1.5,    # chasseur cap 0°
#     xe=0, ye=0, cape=45, vite=1.0     # éviteur devant, fuyant plus vite que chasseur
# )
# print("Résultat :", res3)

# # Cas 4 : éviteur plus rapide que le chasseur — interception impossible
# print("\nTest 4 : Éviteur plus rapide")
# res3 = chasseur_hard(
#     xc=0, yc=0, capc=0, vitc=1.0,    # chasseur cap 0°
#     xe=5, ye=0, cape=0, vite=2.0     # éviteur devant, fuyant plus vite que chasseur
# )
# print("Résultat :", res3)


# Cas 4 : éviteur plus rapide que le chasseur — interception impossible
print("\nTest 4 : Comparaison hard / moyen")
res_m = chasseur_moyen_3(
    xc=800, yc=800, capc=225, vitc=10.0,    # chasseur cap 0°
    xe=100, ye=100, cape=45, vite=10.0     # éviteur devant, fuyant plus vite que chasseur
)
res_h = chasseur_hard(
    xc=800, yc=800, capc=225, vitc=10.0,    # chasseur cap 0°
    xe=100, ye=100, cape=45, vite=10.0     # éviteur devant, fuyant plus vite que chasseur
)
print("Moyen :", res_m)
print("Hard :", res_h)


res_v = chasseur_hard(
    xc=750, yc=500, capc=0, vitc=10.0,    # chasseur cap 0°
    xe=250, ye=250, cape=0, vite=10.0     # éviteur devant, fuyant plus vite que chasseur
)
