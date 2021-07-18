import numpy as np
import json
"""

                 "roPlayerSelection": formatted_playerSelection[0: 10],
                 "roLimbPositionsYou": formatted_limbPositionsYou[0: 10],
                 "roLimbPositionsEnemy": formatted_limbPositionsEnemy[0: 10],
                 "roLimbVelocitiesYou": formatted_limbVelocitiesYou[0: 10],
                 "roLimbVelocitiesEnemy": formatted_limbVelocitiesEnemy[0: 10],
                 "roJointStatesYou": formatted_jointStatesYou[0: 10],
                 "roJointStatesEnemy": formatted_jointStatesEnemy[0: 10],
                 "roGroinRotationsYou": formatted_groinRotationsYou[0: 10],
                 "roGroinRotationsEnemy": formatted_groinRotationsEnemy[0: 10],
                 "roInjuries": formatted_injuries[0: 10],
"""
class DataInputEncoder(json.JSONEncoder):
        def default(self, o):
            params = {
                "limbPositions": o.limb_positions.tolist(),
                "limbVelocities": o.limb_velocities.tolist(),
                "groinRotations": o.groin_rotations.tolist(),
                "jointStates": o.joint_states.tolist(),
                "injuries": o.injuries.tolist(),
                "winner": o.winner,
            }
            return params

class DataInput:
    def __len__(self):
        return len(self.injuries)

    def __init__(self, **kwargs):
        self.limb_positions = np.array(kwargs["limbPositions"], np.float)
        self.limb_velocities = np.array(kwargs["limbVelocities"], np.float)
        self.groin_rotations = np.array(kwargs["groinRotations"], np.float)
        self.joint_states = np.array(kwargs["jointStates"], np.float)
        self.injuries = np.array(kwargs["injuries"], np.float)
        self.winner = kwargs["winner"]
        self.encode = kwargs

class DataOutput:
    def __init__(self):
        pass