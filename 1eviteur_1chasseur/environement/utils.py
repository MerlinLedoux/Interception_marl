import math
import numpy as np

def comp_cap(player_position, target_position):
    dx = target_position[0] - player_position[0]
    dy = target_position[1] - player_position[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return (angle_deg+360)%360

def angle_entre_cap_and_enemy(player_pos, enemy_pos, player_heading_deg):
    dx, dy = enemy_pos[0] - player_pos[0], enemy_pos[1] - player_pos[1]
    angle_to_enemy = np.degrees(np.arctan2(dy, dx))
    angle = (angle_to_enemy - player_heading_deg) % 360
    return angle

def red_dist(distance):
    distance = distance/100
    distance = 10 if distance > 10 else distance
    return distance