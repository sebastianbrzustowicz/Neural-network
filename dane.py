import numpy as np
def dane(data):
  with open(data) as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines)
  return dataset
