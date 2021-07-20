import threading
import time
from collections import deque

from torille import ToribashControl, ToribashSettings
import numpy as np
from  Sources import data as d
from torille.utils import create_random_actions
from Sources import simalution as gs
import json

class DqnState:
    def __init__(self, **kwargs):
        self._values = d.DataInput(**kwargs)
        self._rewards = 0.0

    def get_injuries(self):
        return self._values.injuries[len(self._values.injuries) - 1]

    def dump(self):
        print(json.dumps(self._values, cls=d.DataInputEncoder, indent=4, sort_keys=True))


class DqnSimulation:
    def __init__(self, **kwargs):
        self.game_setting = kwargs["gameSetting"]
        self.pre_sequence = kwargs["preSequence"]
        self.encode = kwargs["encode"]
        self.game = None
        self.winner = -1
        self.sim = threading.Thread(target=self._sim)
        self.sim.start()

    def normalizeEnvStateJoint(self, state):
        state.joint_states = [[1.0 if y == 0 else y for y in x] for x in state.joint_states]
        return state

    def _sim(self):
        self.game = ToribashControl(settings=self.game_setting, draw_game=self.encode["DrawSim"])
        self.game.init()

        post_sequence = []
        for x in range(1, len(self.pre_sequence)):
            state, finish = self.game.get_state()
            if finish:
                self.game.reset()
                return
            self.game.make_actions(self.pre_sequence[x]._values.joint_states.tolist())
        while True:
            state, finish = self.game.get_state()
            if finish:
                self.winner = state.winner
                self.game.reset()
                break
            state = self.normalizeEnvStateJoint(state)
            brainState = DqnState(
                limbPositions=state.limb_positions,
                limbVelocities=state.limb_velocities,
                groinRotations=state.groin_rotations,
                jointStates=state.joint_states,
                injuries=state.injuries,
                winner=state.winner
            )
            post_sequence.append(brainState)
            if not finish:
                if 0.85 > np.random.uniform(0, 1):
                    self.game.make_actions(create_random_actions())
                elif 0.95 > np.random.uniform(0, 1):
                    self.game.make_actions([create_random_actions()[0], [3 for x in range(20)] + [1, 1]])
                elif 0.90 > np.random.uniform(0, 1):
                    self.game.make_actions([[3 for x in range(20)] + [1, 1], create_random_actions()[0]])
                else:
                    self.game.make_actions([[3 for x in range(20)] + [1, 1], [3 for x in range(20)] + [1, 1]])
        self.game.close()
        self.current_sequence = self.pre_sequence + post_sequence

class DQN:
    def __init__(self, **kwargs):
        self._brain = kwargs["Brain"]
        self._epsilon = kwargs["Epsilon"]
        self._repsilon = kwargs["rEpsilon"]
        self._mepsilon = kwargs["mEpsilon"]
        self.encode = kwargs
        self.reset()
        self.fbatch = [deque(maxlen=kwargs["MaxMemory"]), deque(maxlen=kwargs["MaxMemory"]),
                       deque(maxlen=kwargs["MaxMemory"]), deque(maxlen=kwargs["MaxMemory"]),
                       deque(maxlen=kwargs["MaxMemory"]), deque(maxlen=kwargs["MaxMemory"]),
                       deque(maxlen=kwargs["MaxMemory"]), deque(maxlen=kwargs["MaxMemory"]),
                       deque(maxlen=kwargs["MaxMemory"]), deque(maxlen=kwargs["MaxMemory"])]

    # Utils

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

    #

    def chooseAction(self):
        if self._epsilon > np.random.uniform(0, 1):
            self._epsilon *= self._repsilon
            return create_random_actions()
        elif self._epsilon < self._mepsilon:
            self._epsilon = self._mepsilon

        print(self._epsilon)

        #Brain

        prediction = self._brain.predict(self.getEnvPlayerIndex(),
                                         self._sequences_cgame)

        yJoint = np.rint(prediction[len(prediction) - 2][0][0])
        eJoint = np.rint(prediction[len(prediction) - 1][0][0])
        #eJoint = np.rint([3 for x in range(20)] + [1, 1])

        yJoint[len(yJoint) - 1] = 1
        yJoint[len(yJoint) - 2] = 1
        eJoint[len(eJoint) - 1] = 1
        eJoint[len(eJoint) - 2] = 1

        if self.getEnvPlayerIndex()["PlayerID"] == 0:
            action = [eJoint.tolist(), yJoint.tolist()]
        else:
            action = [yJoint.tolist(), eJoint.tolist()]
        print(action)
        try:
            for x in range(len(yJoint)):
                if action[0][x] > 4 or action[0][x] < 1 or action[1][x] > 4 or action[1][x] < 1:
                    raise ValueError("Input Error")
            return action
        except ValueError as v:
            print(v)
            return create_random_actions()

    def isWinAndDraw(self):
        return self._sequences_cgame[len(self._sequences_cgame) - 1]._values.winner == self.encode["PlayerID"] or\
               self._sequences_cgame[len(self._sequences_cgame) - 1]._values.winner == 0

    def simulation(self, finish, it):
        #gs.ISimulation.normalizedRewards(None, PlayerID=self.encode["PlayerID"], EnemyID=self.encode["EnemyID"],
        #                                 GameSequence=self._sequences_cgame)
        if finish and not self.isWinAndDraw():
            return
        elif finish and self.isWinAndDraw():
            fdata = self.encode["Brain"].convertNoDataToAiData(self.getEnvPlayerIndex(), self._sequences_cgame)
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
            #self.encode["Brain"].train(self.fbatch)
            return

        #

        sim_size = self.encode["SimulationSize"]
        sims = []

        #

        settings = ToribashSettings()
        settings.set("custom_settings", 0)
        settings.set("mod", self.encode["ModName"])
        settings.validate_settings()

        #

        for x in range(sim_size):
            sim = DqnSimulation(gameSetting=settings, preSequence=self._sequences_cgame, encode=self.encode)
            sims.append(sim)
        for x in sims:
            x.sim.join()

        #

        playerIndex = self.getToribashPlayerIndex()
        _filter_ = [x for x in sims if x.winner == playerIndex["PlayerID"]]

        #

        for x in _filter_:
            fdata = self.encode["Brain"].convertNoDataToAiData(self.getEnvPlayerIndex(), x.current_sequence)
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
        if len(self.fbatch[0]) != 0 and self.encode["SimulationTrainBegin"] < it:
            self.encode["Brain"].train(self.fbatch)


    def reset(self):
        self._sequences_cgame = []

    def preData(self, **kwargs):
        self._sequences_cgame.append(kwargs["State"])

    def postData(self, **kwargs):
        pass

