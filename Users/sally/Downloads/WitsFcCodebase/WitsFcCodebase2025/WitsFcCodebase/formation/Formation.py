import numpy as np

def GenerateBasicFormation():


    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-7, -2]),  # Left Defender
        np.array([-0, 3]),   # Right Defender
        np.array([7, 1]),    # Forward Left
        np.array([12, 0])      # Forward Right
    ]



    # formation = [
    #     np.array([-13, 0]),    # Goalkeeper
    #     np.array([-10, -2]),  # Left Defender
    #     np.array([-11, 3]),   # Center Back Left
    #     np.array([-8, 0]),    # Center Back Right
    #     np.array([-3, 0]),   # Right Defender
    #     np.array([0, 1]),    # Left Midfielder
    #     np.array([2, 0]),    # Center Midfielder Left
    #     np.array([3, 3]),     # Center Midfielder Right
    #     np.array([8, 0]),     # Right Midfielder
    #     np.array([9, 1]),    # Forward Left
    #     np.array([12, 0])      # Forward Right
    # ]

    return formation
