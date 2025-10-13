import numpy as np

def role_assignment(teammate_positions, formation_positions): 

    # Input : Locations of all teammate locations and positions
    # Output : Map from unum -> positions
    #-----------------------------------------------------------#

    # Example
    point_preferences = {}
    for i in range(1, 6):
        point_preferences[i] = formation_positions[i-1]


    return point_preferences