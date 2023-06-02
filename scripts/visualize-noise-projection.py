import argparse, os, sys, json
# add the root folder of the project to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # for 3d plots magic
import cv2

if '__main__' == __name__:
  parser = argparse.ArgumentParser(description='Visualize the noise projection')
  parser.add_argument(
    '--output', type=str, help='Path to the output file',
    default='docs/img/diffusion-restorator/noise-projection.png'
  )
  args = parser.parse_args()
  
  # Set up the figure and axes
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Center of the large sphere (x0)
  x0 = [0, 0, 0]
  # point x0 with text
  ax.scatter(x0[0], x0[1], x0[2], color='black')
  ax.text(x0[0], x0[1], x0[2], 'x0')

  R = 1.5
  # Generate data for the large sphere
  u = np.linspace(0, 2 * np.pi, 100)
  v = np.linspace(0, np.pi, 100)
  x_large = x0[0] + R * np.outer(np.cos(u), np.sin(v))
  y_large = x0[1] + R * np.outer(np.sin(u), np.sin(v))
  z_large = x0[2] + R * np.outer(np.ones(np.size(u)), np.cos(v))

  # Plot the large sphere
  ax.plot_surface(x_large, y_large, z_large, color='blue', alpha=0.2)
  # as a wireframe
  ax.plot_wireframe(x_large, y_large, z_large, color='blue', alpha=0.2)

  # random point on the large sphere
  centerIndex = np.random.randint(0, len(x_large))
  centerIndex = 23
  x = [x_large[centerIndex, centerIndex], y_large[centerIndex, centerIndex], z_large[centerIndex, centerIndex]]
  # point x with text
  ax.scatter(x[0], x[1], x[2], color='black')
  ax.text(x[0], x[1], x[2], 'x')

  r = 0.5
  # Generate data for the surrounding sphere
  x_surrounding = x[0] + r * np.outer(np.cos(u), np.sin(v))
  y_surrounding = x[1] + r * np.outer(np.sin(u), np.sin(v))
  z_surrounding = x[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
  # Plot the surrounding sphere
  ax.plot_surface(x_surrounding, y_surrounding, z_surrounding, color='red', alpha=0.1)

  # line between x0 and x, red, with text "R"
  ax.plot([x0[0], x[0]], [x0[1], x[1]], [x0[2], x[2]], color='red')
  ax.text((x0[0] + x[0]) / 2, (x0[1] + x[1]) / 2, (x0[2] + x[2]) / 2, 'R')

  # find the points of the large sphere that are in the surrounding sphere
  points = np.array([x_large.flatten(), y_large.flatten(), z_large.flatten()]).T
  points = points[np.linalg.norm(points - x, axis=1) < r]
  # plot the points
  ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', alpha=0.92)
  
  # Set plot limits and labels
  ax.set_xlim(.2, 1.5)
  ax.set_ylim(.2, 1.5)
  ax.set_zlim(.2, 1.5)
  # hide the axes
  ax.set_axis_off()
  # Save the figure
  plt.savefig(args.output, bbox_inches='tight')