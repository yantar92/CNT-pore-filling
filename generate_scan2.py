import itertools
import hashlib
import sys
import numpy as np

def generate_voltage_points():
    """20 voltage points with higher density in 0-0.1V."""
    low = np.linspace(0, 0.1, 10, endpoint=True)      # 10 points
    mid = np.linspace(0.1, 1, 5, endpoint=False)      # 5 points (0.1 already in low)
    high = np.linspace(1, 4, 5, endpoint=True)        # 5 points
    voltages = np.unique(np.concatenate([low, mid, high]))
    return voltages


defect_probabilities = [0, 0.174, 0.5]
temperatures = list(np.arange(298, 500, 50)) + list(np.arange(500, 2000, 500))
radiuses = [5, 10, 15, 20, 30]
voltages = generate_voltage_points()
commands = []
for voltage, prob, temp, radius in itertools.product(voltages, defect_probabilities, temperatures, radiuses):
    param_str = f"{voltage:.3f}_{prob:.3f}_{temp:.0f}_{radius}"
    for rep in range(15):
        seed = hashlib.md5((param_str + f"_{rep}").encode()).hexdigest()
        seed_int = int(seed[:8], 16)
        cmd = (f"python mc-pore.py --voltage {voltage}  --radius {radius} "
               f"--defect_probability {prob} --file {param_str}_{rep}.pkl "
               f"--steps 1000000 --seed {seed_int} --temp {temp}")
        commands.append(cmd)

with open('commands.txt', 'w') as f:
    f.write('\n'.join(commands))
print(f"# To run: cat commands.txt | parallel -j 96", file=sys.stderr)
