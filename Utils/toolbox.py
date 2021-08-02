from torille import ToribashControl, ToribashSettings
from torille.utils import create_random_actions, constants

from torille import ToribashControl, ToribashSettings
from torille.utils import create_random_actions, constants
from Sources import data as d
from Sources import  brain as b
from Sources import dqn

def basic_settings():
    settings = ToribashSettings()
    settings.set("custom_settings", 0)
    settings.set("mod", "lenshu3ng.tbm")
    settings.validate_settings()
    return settings

BASIC = basic_settings()


def convert_replay(sreplay):
    s = [
        dqn.DqnState(
            limbPositions=_state_.limb_positions,
            limbVelocities=_state_.limb_velocities,
            groinRotations=_state_.groin_rotations,
            jointStates=_state_.joint_states,
            injuries=_state_.injuries,
            winner=_state_.winner
        ) for _state_ in sreplay
    ]
    return s


def get_single_only_mod(r, mod):
    v = None
    with open(r, "r") as fd:
        lines = fd.readlines()
        for l in lines:
            if mod in l:
                v = r
    return  v

def get_only_mod(all_replay, mod):
    a = []
    for r in all_replay:
        with open(r, "r") as fd:
            lines = fd.readlines()
            for l in lines:
                if mod in l:
                    a.append(r)
    return a