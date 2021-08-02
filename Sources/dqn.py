import threading
import time
import os
from collections import deque
from Utils import toolbox
from torille import ToribashControl, ToribashSettings
import numpy as np
from  Sources import data as d
from torille.utils import create_random_actions
from Sources import simalution as gs
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
                if 0.80 > np.random.uniform(0, 1):
                    self.game.make_actions(create_random_actions())
                elif 0.80 > np.random.uniform(0, 1):
                    self.game.make_actions([create_random_actions()[0], [3 for x in range(20)] + [1, 1]])
                elif 0.80 > np.random.uniform(0, 1):
                    self.game.make_actions([[3 for x in range(20)] + [1, 1], create_random_actions()[0]])
                else:
                    self.game.make_actions([[3 for x in range(20)] + [1, 1], [3 for x in range(20)] + [1, 1]])
        self.game.close()
        self.current_sequence = self.pre_sequence + post_sequence

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
        self.reset()
        self.fbatch = [[], [],
                       [], [],
                       [], [],
                       [], [],
                       [], []]
        self.fbatch_mutex = threading.Lock()
        if self.encode["ConvertReplay"]:
            self._replay_thread = threading.Thread(target=self.read_save_replay)
            self._replay_thread.start()
            if self.encode["JoinReplay"]:
                self._replay_thread.join()

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

    def read_save_replay(self):
        path = self.encode["ReplayData"]
        all_replay = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and toolbox.get_single_only_mod(os.path.join(path, f), "lenshu3ng.tbm")]

        toribash = ToribashControl(settings=toolbox.BASIC, draw_game=True)
        toribash.init()
        toribash.finish_game()

        for r in all_replay:
            json_path = os.path.join(path, r).replace(".rpl", ".json")
            if not os.path.isfile(json_path):
                s = toribash.read_replay("autosave/"+r)
                x = toolbox.convert_replay([ self.normalizeEnvStateJoint(k) for k in s])
                winner = x[ilen(x)]._values.winner
                if winner == 0:
                    continue
                playerId = 2 if winner == 2 else 1
                enemyId = 1 if winner == 1 else 2
                self.fbatch_mutex.acquire()
                self.pushFbatch(self.getCustomEnvPlayerIndex(playerId, enemyId), x)
                self.fbatch_mutex.release()
                with open(json_path, "w") as fd:
                    fd.write(json.dumps(x, cls=DqnStateEncode))
        toribash.close()
    # Utils

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
        """
        self.fbatch[0].append([fdata[0][x] if x < len(fdata[0]) else np.array([0 for _ in range(len(fdata[0][0]))]) for x in range(15)])
        self.fbatch[1].append([fdata[1][x] if x < len(fdata[1]) else np.array([0 for _ in range(len(fdata[1][0]))]) for x in range(15)])
        self.fbatch[2].append([fdata[2][x] if x < len(fdata[2]) else np.array([0 for _ in range(len(fdata[2][0]))]) for x in range(15)])
        self.fbatch[3].append([fdata[3][x] if x < len(fdata[3]) else np.array([0 for _ in range(len(fdata[3][0]))]) for x in range(15)])
        self.fbatch[4].append([fdata[4][x] if x < len(fdata[4]) else np.array([0 for _ in range(len(fdata[4][0]))]) for x in range(15)])
        self.fbatch[5].append([fdata[5][x] if x < len(fdata[5]) else np.array([0 for _ in range(len(fdata[5][0]))]) for x in range(15)])
        self.fbatch[6].append([fdata[6][x] if x < len(fdata[6]) else np.array([0 for _ in range(len(fdata[6][0]))]) for x in range(15)])
        self.fbatch[7].append([fdata[7][x] if x < len(fdata[7]) else np.array([0 for _ in range(len(fdata[7][0]))]) for x in range(15)])
        self.fbatch[8].append([fdata[8][x] if x < len(fdata[8]) else np.array([0 for _ in range(len(fdata[8][0]))]) for x in range(15)])
        self.fbatch[9].append([fdata[9][x] if x < len(fdata[9]) else np.array([0 for _ in range(len(fdata[9][0]))]) for x in range(15)])
        """
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

    #

    def chooseAction(self):
        if self._epsilon > np.random.uniform(0, 1):
            self._epsilon *= self._repsilon
            print(self._epsilon)
            return create_random_actions()
        elif self._epsilon < self._mepsilon:
            self._epsilon = self._mepsilon

        predictionAIPlayer = self._brain.predict2(self.getEnvPlayerIndex(),
                                          self._sequences_cgame)
        predictionAIEnemy = self._brain.predict2(self.getCustomEnvPlayerIndex(1, 2),
                                          self._sequences_cgame)
        yJoint = np.rint(predictionAIPlayer["roJointStatesYou"][0][0])
        eJoint = np.rint(predictionAIEnemy["roJointStatesYou"][0][0])

        yJoint[len(yJoint) - 1] = 1
        yJoint[len(yJoint) - 2] = 1
        eJoint[len(eJoint) - 1] = 1
        eJoint[len(eJoint) - 2] = 1
        #Brain

        if self.getEnvPlayerIndex()["PlayerID"] == 0:
            action = [eJoint.tolist(), yJoint.tolist()]
        else:
            action = [yJoint.tolist(), eJoint.tolist()]
        print(action)
        """
        prediction = self._brain.predict(self.getEnvPlayerIndex(),
                                         self._sequences_cgame)

        yJoint = np.rint(prediction[len(prediction) - 2][0][0])
        #eJoint = np.rint(prediction[len(prediction) - 1][0][0])
        eJoint = np.rint([3 for x in range(20)] + [1, 1])

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
                if action[0][x] > 4 or action[0][x] < 1:
                    raise ValueError("Input Error")
                elif  action[1][x] > 4 or action[1][x] < 1:
                    raise ValueError("Input Error")
            return action
        except ValueError as v:
            print(v)
            return create_random_actions()
        """
        try:
            for x in range(len(yJoint)):
                if action[0][x] > 4 or action[0][x] < 1:
                    raise ValueError("Input Error")
                elif action[1][x] > 4 or action[1][x] < 1:
                    raise ValueError("Input Error")
            return action
        except ValueError as v:
            print(v)
            return create_random_actions()


    def isWinAndDraw(self):
        return self._sequences_cgame[len(self._sequences_cgame) - 1]._values.winner == self.encode["PlayerID"] or\
               self._sequences_cgame[len(self._sequences_cgame) - 1]._values.winner == 0

    def isWin(self):
        return self._sequences_cgame[len(self._sequences_cgame) - 1]._values.winner == self.encode["PlayerID"]

    def simulation(self, finish, it):
        #gs.ISimulation.normalizedRewards(None, PlayerID=self.encode["PlayerID"], EnemyID=self.encode["EnemyID"],
        #                                 GameSequence=self._sequences_cgame)
        if finish and not self.isWin():
            fdata = self.encode["Brain"].convertNoDataToAiData(self.getCustomEnvPlayerIndex(1, 2), self._sequences_cgame)
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
            self.encode["Brain"].train2([self.fbatch[x][ilen(self.fbatch[x])-1:ilen(self.fbatch[x])] for x in range(len(self.fbatch))], self.encode)
            return
        elif finish and self.isWin():
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

            self.encode["Brain"].train2([self.fbatch[x][ilen(self.fbatch[x])-1:ilen(self.fbatch[x])] for x in range(len(self.fbatch))], self.encode)
            return
            #bsize = int(len(self.fbatch[0]) / 16)
            #for it in range(bsize):
            #    self.encode["Brain"].train([ self.fbatch[x][it : it + 16] for x in range(len(self.fbatch))], self.encode)
            #return

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
            bsize = int(len(self.fbatch[0]) / 16)
            for it in range(bsize):
                self.encode["Brain"].train([ self.fbatch[x][it : it + 16] for x in range(len(self.fbatch))], self.encode)


    def reset(self):
        self._sequences_cgame = []

    def preData(self, **kwargs):
        self._sequences_cgame.append(kwargs["State"])

    def postData(self, **kwargs):
        pass

