from SourcesPaperspace import data as d
from SourcesPaperspace import toolbox

import threading
import os
import json


def ilen(a): return len(a) - 1

class DqnState:
    def __init__(self, **kwargs):
        self._values = d.DataInput(**kwargs)
        self._rewards = 0.0

    def get_injuries(self):
        return self._values.injuries[len(self._values.injuries) - 1]

    def dump(self):
        print(json.dumps(self._values, cls=d.DataInputEncoder, indent=4, sort_keys=True))

class DqnStateEncode(json.JSONEncoder):
    def default(self, o):
        params = {
            "limbPositions": o._values.limb_positions.tolist(),
            "limbVelocities": o._values.limb_velocities.tolist(),
            "groinRotations": o._values.groin_rotations.tolist(),
            "jointStates": o._values.joint_states.tolist(),
            "injuries": o._values.injuries.tolist(),
            "winner": o._values.winner,
        }
        return params


class DQN:
    def __init__(self, **kwargs):
        self._brain = kwargs["Brain"]
        self._epsilon = kwargs["Epsilon"]
        self._repsilon = kwargs["rEpsilon"]
        self._mepsilon = kwargs["mEpsilon"]
        self.encode = kwargs
        self.fbatch = [[], [],
                       [], [],
                       [], [],
                       [], [],
                       [], []]
        self.fbatch_mutex = threading.Lock()
        if self.encode["LoadReplay"]:
            self.load_replay()

    def load_replay(self):
        path = self.encode["ReplayData"]
        all_replay = [f for f in os.listdir(path) if
                      os.path.isfile(os.path.join(path, f)) and toolbox.get_single_only_mod(os.path.join(path, f),
                                                                                            "lenshu3ng.tbm")]
        for r in all_replay:
            json_path = os.path.join(path, r).replace(".rpl", ".json")
            if os.path.isfile(json_path):
                with open(json_path, "r") as fd:
                    data = json.load(fd)
                    data = [ data[k] for k in range(len(data)) if k % 30 == 0]
                    if (len(data) > 15):
                        continue
                    print("Loaded Replay " + json_path)
                    x = [ DqnState(
                            limbPositions=_["limbPositions"],
                            limbVelocities=_["limbVelocities"],
                            groinRotations=_["groinRotations"],
                            jointStates=_["jointStates"],
                            injuries=_["injuries"],
                            winner=_["winner"]
                            )
                            for _ in data
                        ]
                    winner = x[ilen(x)]._values.winner
                    playerId = 2 if winner == 2 else 1
                    enemyId = 1 if winner == 1 else 2
                    self.fbatch_mutex.acquire()
                    self.pushFbatch(self.getCustomEnvPlayerIndex(playerId, enemyId), x)
                    self.fbatch_mutex.release()

    def pushFbatch(self, playerEnvIndex, s):
        fdata = self.encode["Brain"].convertNoDataToAiData(playerEnvIndex, s)

        self.fbatch[0].append(fdata[0])
        self.fbatch[1].append(fdata[1])
        self.fbatch[2].append(fdata[2])
        self.fbatch[3].append(fdata[3])
        self.fbatch[4].append(fdata[4])
        self.fbatch[5].append(fdata[5])
        self.fbatch[6].append(fdata[6])
        self.fbatch[7].append(fdata[7])
        self.fbatch[8].append(fdata[8])
        self.fbatch[9].append(fdata[9])

    def getCustomEnvPlayerIndex(self, playerId, enemyId):
        return\
        {
            "PlayerID": 0 if playerId == 2 else 1,
            "EnemyID": 1 if enemyId == 1 else 0,
        }

    def getToribashPlayerIndex(self):
        return\
        {
            "PlayerID": self.encode["PlayerID"],
            "EnemyID": self.encode["EnemyID"],
        }

    def getEnvPlayerIndex(self):
        return\
        {
            "PlayerID": 0 if self.encode["PlayerID"] == 2 else 1,
            "EnemyID": 1 if self.encode["EnemyID"] == 1 else 0,
        }

    def normalizeEnvStateJoint(self, state):
        state.joint_states = [[1.0 if y == 0 else y for y in x] for x in state.joint_states]
        return state