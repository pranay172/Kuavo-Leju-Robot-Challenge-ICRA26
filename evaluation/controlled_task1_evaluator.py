"""Apply explicit identity/side controls to the original Task 1 evaluator."""
import json
from pathlib import Path
import runpy


class FixedSideRandom:
    def __init__(self, original, coin):
        self.original = original
        self.coin = coin

    def choice(self, values):
        return self.coin if list(values) == [0, 1] else self.original.choice(values)

    def __getattr__(self, name):
        return getattr(self.original, name)


def run(original_script, seed, control, receipt):
    namespace = runpy.run_path(original_script)
    task = namespace['SimulatorTask1'](seed)
    if control['toy'] not in task.toy_obj or control['car'] not in task.car_obj:
        raise ValueError('Controlled identities must be existing Task 1 objects')
    side = control['animal_side']
    if side not in ['viewer_left', 'viewer_right']:
        raise ValueError('Unknown controlled starting side')
    task.toy, task.car = control['toy'], control['car']
    x, y = float(control['x']), float(control['abs_y'])
    if not .23 <= x <= .43 or not .22 <= y <= .45:
        raise ValueError('Fixed positions must remain within the official spawn regions')
    for region, sign in [(task.left_spawn_region, 1), (task.right_spawn_region, -1)]:
        region['x'] = [x, x]
        region['y'] = [sign*y, sign*y]
    coin = 0 if side == 'viewer_left' else 1
    task.rng = FixedSideRandom(task.rng, coin)
    Path(receipt).write_text(json.dumps({'seed': seed, 'controlled': True, 'toy': task.toy,
        'car': task.car, 'animal_side': side, 'left_spawn_region': task.left_spawn_region,
        'right_spawn_region': task.right_spawn_region,
        'toy_destination': task.toy_box_region, 'car_destination': task.car_box_region,
        'scoring': 'Original evaluator and original regions unchanged.'}, indent=2) + '\n')
    task.run()
