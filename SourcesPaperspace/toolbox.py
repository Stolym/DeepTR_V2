from SourcesPaperspace import dqn


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