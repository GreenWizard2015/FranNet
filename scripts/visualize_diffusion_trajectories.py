# this script visualizes the trajectories of the fixed points of the diffusion process for different beta schedules
# NOT diffusion process during inference
import os, sys
# add the root folder of the project to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from NN.restorators.diffusion.diffusion_schedulers import CDPDiscrete, get_beta_schedule
from NN.restorators.diffusion.diffusion import CGaussianDiffusion
import numpy as np

# argparsing
import argparse
parser = argparse.ArgumentParser(description='Visualize diffusion trajectories')
parser.add_argument('--fix-beta-end', action='store_true', help='Fix the end of the beta schedule to 1.0', default=False)
parser.add_argument('--npoints', type=int, help='Number of points to visualize', default=6)
parser.add_argument('--seed', type=int, help='Random seed', default=11)
parser.add_argument('--save', action='store_true', help='Save the plot', default=False)

args = parser.parse_args()
fixBetaEnd = args.fix_beta_end
NPoints = args.npoints
seed = args.seed

# make the beta schedule run to 1.0 instead of their defaults
def _getBetaSchedule(beta_schedule):
  scheduler = get_beta_schedule(beta_schedule)
  if not fixBetaEnd: return lambda: scheduler
  def fixed(s): return scheduler(s, beta_end=0.9999)
  return lambda: fixed

schedulers = [
  lambda: get_beta_schedule('cosine'),
  _getBetaSchedule('linear'),
  _getBetaSchedule('quadratic'),
  _getBetaSchedule('sigmoid'),
]
plotArgsList = [
  {'label': 'cosine', 'linestyle': '-'},
  {'label': 'linear', 'linestyle': '--'},
  {'label': 'quadratic', 'linestyle': '-.'},
  {'label': 'sigmoid', 'linestyle': ':'},
]
fig, ax = plt.subplots(1, 1)

pointsAB = tf.random.normal((2, NPoints, 1), seed=seed)
# plot the initial points
line = plt.plot(np.full((NPoints, 1), -1).transpose(), pointsAB[0].numpy().transpose(), 'o')
# collect the colors of the initial points
colors = [l.get_color() for l in line]

line = plt.plot(np.full((NPoints, 1), 100 - 1).transpose(), pointsAB[1].numpy().transpose(), 'o')
# assign the colors of the initial points to the final points
for l, c in zip(line, colors): l.set_color(c)

for scheduler, plotArgs in zip(schedulers, plotArgsList):
  diff = CGaussianDiffusion(
    channels=1,
    schedule=CDPDiscrete(
      beta_schedule=scheduler(),
      noise_steps=100,
      t_schedule=None,
    ),
    sampler=None,
    lossScaling=None
  )
  pointsA = pointsAB[0]
  pointsB = pointsAB[1]

  NSteps = diff._schedule.noise_steps
  # make (NPoints * NSteps, 1) by repeating each point NSteps times
  pointsA = tf.repeat(pointsA, NSteps, axis=0)
  pointsB = tf.repeat(pointsB, NSteps, axis=0)

  # continuous time tensor
  t = tf.range(NSteps, dtype=tf.int32)
  t = tf.reshape(t, (1, NSteps))
  t = tf.tile(t, (NPoints, 1))
  t = tf.reshape(t, (NPoints * NSteps, 1))

  # make the diffusion process
  points = diff._forwardStep(pointsA, pointsB, t)['xT']
  points = tf.reshape(points, (NPoints, NSteps, 1))

  # plot the diffusion process
  t = t.numpy().reshape((NPoints, NSteps, 1))
  points = points.numpy()
  # add t -1 and initial point
  t = np.concatenate((np.full((NPoints, 1, 1), -1), t), axis=1)
  points = np.concatenate((pointsAB[0].numpy().reshape((NPoints, 1, 1)), points), axis=1)
  # draw the lines for each trajectory
  for i in range(NPoints):
    clr = colors[i]
    line = ax.plot(t[i], points[i], color=clr, **plotArgs)
  continue

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
for lbl, handle in by_label.items():
  # make a copy of the handle
  new_handle = matplotlib.lines.Line2D([0], [0], linestyle=handle.get_linestyle(), color='black')
  by_label[lbl] = new_handle
  continue
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

ax.set_xlabel('t')
ax.set_ylabel('x')
# all margins are 1% of the plot size
plt.margins(0.01)
fig.set_size_inches(2 * fig.get_size_inches())
plt.tight_layout()

if args.save:
  plt.savefig(args.save)
else:
  plt.show()