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

    def _sim(self):
        self.game = ToribashControl(settings=self.game_setting, draw_game=False)
        self.game.init()

        finish = False
        post_sequence = []
        for x in range(len(self.pre_sequence)):
            state, finish = self.game.get_state()
            self.game.make_actions(self.pre_sequence[x]._values.joint_states.tolist())

        while not finish:
            state, finish = self.game.get_state()
            if finish:
                self.winner = state.winner
                self.game.reset()
                break
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
                self.game.make_actions(create_random_actions())
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
        self.memory = deque(maxlen=1000)

    def chooseAction(self):
        if self._epsilon > np.random.uniform(0, 1):
            self._epsilon *= self._repsilon
            return create_random_actions()
        elif self._epsilon < self._mepsilon:
            self._epsilon = self._mepsilon
        print(self._epsilon)

        #Brain
        if len(self._sequences_cgame) == 0:
            yJoint = np.rint([3 for x in range(20)] + [1, 1])
            eJoint = np.rint([3 for x in range(20)] + [1, 1])
        else:
            prediction = self._brain.predict(self.encode["PlayerID"],
                                             self._sequences_cgame)
            yJoint = np.rint(prediction[len(prediction) - 2][0][0])
            eJoint = np.rint(prediction[len(prediction) - 1][0][0])
            yJoint[len(yJoint) - 1] = 1
            yJoint[len(yJoint) - 2] = 1
            eJoint[len(eJoint) - 1] = 1
            eJoint[len(eJoint) - 2] = 1
        action = [eJoint.tolist(), yJoint.tolist()]
        print(action)
        try:
            for x in range(len(yJoint)):
                if action[0][x] > 4 or action[0][x] < 1 or action[1][x] > 4 or action[1][x] < 1:
                    raise ValueError("Input Error");
            return action
        except ValueError as v:
            print(v)
            return create_random_actions()

    def isWinAndDraw(self):
        return self._sequences_cgame[len(self._sequences_cgame) - 1]._values.winner - 1 == self.encode["PlayerID"] or self._sequences_cgame[len(self._sequences_cgame) - 1]._values.winner == 0
        #return self._sequences_cgame[len(self._sequences_cgame) - 1]._values.injuries[self.encode["PlayerID"]] > \
        #       self._sequences_cgame[len(self._sequences_cgame) - 1]._values.injuries[self.encode["EnemyID"]]

    def _simulation(self, finish):
        gs.ISimulation.normalizedRewards(None, PlayerID=self.encode["PlayerID"], EnemyID=self.encode["EnemyID"],
                                         GameSequence=self._sequences_cgame)
        if finish and not self.isWinAndDraw():
            return
        elif finish and self.isWinAndDraw():
            fbatch = [[],[],[],[],[],[],[],[],[],[]]
            fdata = self.encode["Brain"].convertNoDataToAiData(self.encode["PlayerID"], self._sequences_cgame)
            fbatch[0].append(fdata[0])
            fbatch[1].append(fdata[1])
            fbatch[2].append(fdata[2])
            fbatch[3].append(fdata[3])
            fbatch[4].append(fdata[4])
            fbatch[5].append(fdata[5])
            fbatch[6].append(fdata[6])
            fbatch[7].append(fdata[7])
            fbatch[8].append(fdata[8])
            fbatch[9].append(fdata[9])
            self.encode["Brain"].train(fbatch)
            return
        # Simulation
        sim_size = 32
        sims = []
        settings = ToribashSettings()
        settings.set("custom_settings", 0)
        settings.set("mod", "lenshu3ng.tbm")
        settings.validate_settings()

        for x in range(sim_size):
            sim = DqnSimulation(gameSetting=settings, preSequence=self._sequences_cgame, encode=self.encode)
            sims.append(sim)
        for x in sims:
            x.sim.join()
        filter = [x for x in sims if x.winner == self.encode["PlayerID"] + 1]
        fbatch = [[],[],[],[],[],[],[],[],[],[]]
        for x in filter:
            fdata = self.encode["Brain"].convertNoDataToAiData(self.encode["PlayerID"], x.current_sequence)
            fbatch[0].append(fdata[0])
            fbatch[1].append(fdata[1])
            fbatch[2].append(fdata[2])
            fbatch[3].append(fdata[3])
            fbatch[4].append(fdata[4])
            fbatch[5].append(fdata[5])
            fbatch[6].append(fdata[6])
            fbatch[7].append(fdata[7])
            fbatch[8].append(fdata[8])
            fbatch[9].append(fdata[9])
        if (len(fbatch[0]) != 0):
            self.encode["Brain"].train(fbatch)


    def reset(self):
        self._sequences_cgame = []

    def preData(self, **kwargs):
        self._sequences_cgame.append(kwargs["State"])

    def postData(self, **kwargs):
        pass

