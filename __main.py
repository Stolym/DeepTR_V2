import json
import time

from torille import ToribashControl, ToribashSettings
from torille.utils import create_random_actions, constants
from Sources import data as d
from Sources import  brain as b
from Sources import dqn

# Settings "Rule"

settings = ToribashSettings()
settings.set("custom_settings", 0)
settings.set("mod", "lenshu3ng.tbm")
settings.validate_settings()

#

toribash = ToribashControl(settings=settings, draw_game=True)
toribash.init()

# Red Array 1 Winner Index 1
    # Blue Array 0 Winner Index 2

# Analyst Mod #

_iterator_ = 0
_end_ = 10000

_brain_ = b.Brain()
_dqn_ = dqn.DQN(Brain=_brain_, DrawSim=False, ModName="lenshu3ng.tbm", SimulationTrainBegin=5, SimulationSize=16, MaxMemory=1000,
                PlayerID=2, EnemyID=1, Epsilon=0.03, rEpsilon=0.999, mEpsilon=0.03)

_end_simulation_ = 0

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
        _dqn_.simulation(_finish_, _iterator_)
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