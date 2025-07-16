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

    forward = np.array([np.cos(capc_rad), np.sin(capc_rad)])
    right = np.array([-np.sin(capc_rad), np.cos(capc_rad)])

    ax = np.clip(np.dot(target_vec, forward), -1.0, 1.0)
    ay = np.clip(np.dot(target_vec, right), -1.0, 1.0)  
    
    return np.array([ax, ay])




#----------------Zone de test----------------#

xc, yc, capc = 300, 100, 180 
xe, ye, cape = 100, 100, 45

chasseur_moyen(xc, yc, capc, xe, ye, cape)