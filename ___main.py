import json
import os
import time

from torille import ToribashControl, ToribashSettings
from torille.utils import create_random_actions, constants
from Sources import data as d
from Sources import  brain as b
from Sources import dqn

# Settings "Rule"
from Utils import toolbox
#

#toribash = ToribashControl(settings=toolbox.BASIC, draw_game=True)
#toribash.init()

# Red Array 1 Winner Index 1
# Blue Array 0 Winner Index 2
#Tick Env#
def ilen(a): return len(a) - 1

# Get ALL Replay who is lenshu
path = "C:/Users/node/PycharmProjects/GodUKE/venv/Lib/site-packages/torille/toribash/replay/autosave"
#all_replay_from_ghost = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and toolbox.get_single_only_mod(os.path.join(path, f), "lenshu3ng.tbm")]

# Analyst Mod #
_iterator_ = 0
_end_ = 10000
_end_simulation_ = 0

_brain_ = b.Brain()
_dqn_ = dqn.DQN(Brain=_brain_, NameModel="Model.ckpt", DirModel="Datav4", DrawSim=False, ModName="lenshu3ng.tbm",
                ReplayData=path, ConvertReplay=False, LoadReplay=False, JoinReplay=False,
                SimulationTrainBegin=-1, SimulationSize=4, MaxMemory=10000,
                PlayerID=2, EnemyID=1, Epsilon=0.03, rEpsilon=0.999, mEpsilon=0.03)
#_dqn_.encode["Brain"].advancedTrain(
#    data=_dqn_.fbatch, sequence_size=15, encode=_dqn_.encode, step=10, batch_size=24, epochs=24)


toribash = ToribashControl(settings=toolbox.BASIC, draw_game=True)
toribash.init()

while True:
    _state_, _finish_ = toribash.get_state()
    if _finish_:
        _state_ = _dqn_.normalizeEnvStateJoint(_state_)
        brainState = dqn.DqnState(
            limbPositions=_state_.limb_positions,
            limbVelocities=_state_.limb_velocities,
            groinRotations=_state_.groin_rotations,
            jointStates=_state_.joint_states,
            injuries=_state_.injuries,
            winner=_state_.winner
        )
        _dqn_.preData(State=brainState)
        #_dqn_.simulation(_finish_, _iterator_)
        _state_ = toribash.reset()
        _dqn_.reset()
        _iterator_ += 1
        if _iterator_ == _end_:
            break
    _state_ = _dqn_.normalizeEnvStateJoint(_state_)
    brainState = dqn.DqnState(
        limbPositions=_state_.limb_positions,
        limbVelocities=_state_.limb_velocities,
        groinRotations=_state_.groin_rotations,
        jointStates=_state_.joint_states,
        injuries=_state_.injuries,
        winner=_state_.winner
    )
    _dqn_.preData(State=brainState)
    if _iterator_ < _end_simulation_ and not _finish_:
        _dqn_.simulation(_finish_, _iterator_)
    _action_ = _dqn_.chooseAction()
    toribash.make_actions(_action_)
toribash.close()







