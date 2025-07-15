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


def chasseur_simple(xc, yc, capc, vitc, xe, ye, cape, vite):
    dx = xe - xc
    dy = ye - yc
    dist = np.hypot(dx, dy)
    

    if dist > 1e-3:
        target_ax = 1.0 * (dx / dist)  # force vers l'éviteur
        target_ay = 1.0 * (dy / dist)
    else:
        target_ax, target_ay = 0.0, 0.0

    return(np.array([target_ax, target_ay]))