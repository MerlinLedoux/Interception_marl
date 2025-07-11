import numpy as np
import math

def move(x, y, cap, vit, ax, ay, max_speed):

    ax = ax if vit + ax >= 0.1 else vit - 0.1
    
    x3 = x + np.cos(cap) * (vit + ax) - np.sin(cap) * ay
    y3 = y + np.sin(cap) * (vit + ax) + np.cos(cap) * ay

    new_cap = math.atan2(y3 - y, x3 - x)
    vitesse = np.linalg.norm(np.array([x3 - x, y3 - y]))
    
    new_vit = vitesse if vitesse <= max_speed else max_speed

    new_x = x + np.cos(new_cap) * new_vit
    new_y = y + np.sin(new_cap) * new_vit
    new_cap = (new_cap + 2*np.pi) % (2*np.pi)
    
    return(new_x, new_y, new_cap, new_vit)