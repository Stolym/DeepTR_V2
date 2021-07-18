import json
import time

from torille import ToribashControl, ToribashSettings
from torille.utils import create_random_actions, constants
from Sources import data as d
from Sources import  brain as b
from Sources import dqn

# Show gameplay
settings = ToribashSettings()
settings.set("custom_settings", 0)
settings.set("mod", "lenshu3ng.tbm")
settings.validate_settings()
toribash = ToribashControl(settings=settings, draw_game=True)
toribash.init()

it = 0
__end__ = 10000

state = None
nstate = None
t = False

esim = 20

_brain = b.Brain()
_dqn = dqn.DQN(Brain=_brain, PlayerID=1, EnemyID=0, Epsilon=0.50, rEpsilon=0.999999, mEpsilon=0.1)

# Toribash State
# Player Selection
# Winner

while True:
    #Break Iterator
    if it == __end__:
        break
    #Current State
    if not t:
        state, t = toribash.get_state()
    else:
        t = False
    state.joint_states = [[1 if y == 0 else y for y in x] for x in state.joint_states]
    brainState = dqn.DqnState(
        limbPositions=state.limb_positions,
        limbVelocities=state.limb_velocities,
        groinRotations=state.groin_rotations,
        jointStates=state.joint_states,
        injuries=state.injuries,
        winner=state.winner
    )
    _dqn.preData(State=brainState)
    #End Current Game
    if t:
        _dqn._simulation(t)
        _dqn.reset()
        state = toribash.reset()
        it += 1
    else:
        if (it < esim):
            _dqn._simulation(t)
        action = _dqn.chooseAction()
        toribash.make_actions(action)
toribash.close()
"""
print(constants.NUM_LIMBS)
print(constants.NUM_CONTROLLABLES)
input()
"""