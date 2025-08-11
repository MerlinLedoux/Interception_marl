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

def normaliser_chasseur(observation_chasseur):
    
    cap1_norm = observation_chasseur[0] / 180 - 1
    vit1_norm = observation_chasseur[1] / 5 - 1
    dist1_norm = observation_chasseur[2] / 5 - 1
    cap2_norm = observation_chasseur[3] / 180 - 1
    cap3_norm = observation_chasseur[4] / 180 - 1
    vit2_norm = observation_chasseur[5] / 5 - 1
    dist2_norm = observation_chasseur[6] / 5 - 1
    cap4_norm = observation_chasseur[7] / 180 - 1
    cap5_norm = observation_chasseur[8] / 180 - 1
    vit3_norm = observation_chasseur[9] / 5 - 1
    
    chasseur_norm = np.array([cap1_norm, vit1_norm, dist1_norm, cap2_norm, cap3_norm, vit2_norm, 
                                   dist2_norm, cap4_norm, cap5_norm, vit3_norm], dtype=np.float32)
    
    return chasseur_norm