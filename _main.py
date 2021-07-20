import time
from torille import ToribashControl, ToribashSettings

settings = ToribashSettings()
settings.set("custom_settings", 0)
settings.set("mod", "lenshu3ng.tbm")
settings.validate_settings()
toribash = ToribashControl(settings=settings, draw_game=True)
toribash.init()

# Red Array 1 Winner Index 1
# Blue Array 0 Winner Index 2

finish = False
it = 0
while True:
    state, finish = toribash.get_state()
    if finish:
        state = toribash.reset()
        it += 1
    state.joint_states = [[1.0 if y == 0 else y for y in x] for x in state.joint_states]
    if it == 1:
        break
    toribash.make_actions(state.joint_states)
    time.sleep(0.2)
toribash.close()