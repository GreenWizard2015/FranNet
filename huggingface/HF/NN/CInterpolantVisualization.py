import hashlib, json, os, time, glob, sys
import numpy as np
import math
import imageio
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NN.restorators.samplers.CSamplerWatcher import CSamplerWatcher
from NN.utils import extractInterpolated, ensure4d

from collections import namedtuple
CCollectedSteps = namedtuple('CCollectedSteps', ['value', 'x0', 'x1', 'totalSteps', 'totalPoints'])

def hasharr(arr):
  hash = hashlib.blake2b(arr.tobytes(), digest_size=20)
  for dim in arr.shape:
    hash.update(dim.to_bytes(4, byteorder='big'))
  return hash.hexdigest()

def plot2image(fig):
  canvas = fig.canvas
  ncols, nrows = fig.canvas.get_width_height()
  with io.BytesIO() as buf:
    fig.savefig(buf, format='raw')
    buf.seek(0)
    buffered = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = buffered.reshape(nrows, ncols, -1)
    pass
  M = 16
  pad1 = M - (image.shape[0] % M)
  pad2 = M - (image.shape[1] % M)
  if (pad1 != M) or (pad2 != M):
    padA = padB = (0, 0)
    if pad1 != 0: padA = (math.floor(pad1 / 2), math.ceil(pad1 / 2))
    if pad2 != 0: padB = (math.floor(pad2 / 2), math.ceil(pad2 / 2))
    image = np.pad(image, (padA, padB, (0, 0)), 'constant', constant_values=255)
  return image

def _plotColorCurve2d(collectedSteps, originalColors):
  collectedSteps = collectedSteps.value
  N = collectedSteps.shape[0]
  maxD = np.abs(collectedSteps).max()
  flatColors = collectedSteps.reshape(N, -1)
  pointsN = flatColors.shape[-1]
  flatColorsT = flatColors.T

  def F(step, ax):
    values = collectedSteps[step]
    ax.cla()
    ax.set_title(f'Color trajectories (per channel)')
    colors = []
    # draw the trajectory
    stepsX = np.arange(N)
    for j in range(len(flatColorsT)):
      trajectory = flatColorsT[j]
      x = ax.plot(stepsX, trajectory, linestyle='dotted')
      colors.append(x[0].get_color())
    # draw the current position
    stepsX = np.full((pointsN, ), step)
    ax.plot(stepsX, values.flatten(), 'ro', markersize=3)
    # draw the original colors if present
    if originalColors is not None:
      for j, value in enumerate(originalColors.flatten()):
        ax.plot(N - 1, value, 'o', markersize=2, color=colors[j])
      pass
    ax.set_ylim(-maxD, maxD)
    return
  return F

def _plotEuclideanDistanceCurve2d(collectedSteps, originalColors):
  collectedSteps = collectedSteps.value
  N = collectedSteps.shape[0]
  pointsN = collectedSteps.shape[1]
  targets = originalColors if originalColors is not None else collectedSteps[-1]
  # euclidean distances
  distances = np.sqrt(np.sum((collectedSteps - targets[None]) ** 2, axis=-1)).T
  assert distances.shape == (pointsN, N), 'Unexpected shape of distances'

  def F(step, ax):
    ax.cla()
    ax.set_title(f'Euclidean distance (log scale)')
    # draw the trajectory
    stepsX = np.arange(N)
    for j, distance in enumerate(distances):
      ax.plot(stepsX, distance, linestyle='dotted')

    # draw the current position
    stepsX = np.full((pointsN, ), step)
    ax.plot(stepsX, distances[:, step], 'ro', markersize=3)
    ax.set_yscale('log')
    return
  return F

def _plotRGBTrajectories(Values, originalColors, title, clip=None):
  pointsN = Values.shape[1]
  def F(step, ax):
    values = Values[step]
    ax.cla()
    ax.set_title(title)
    colors = []
    # draw the trajectory
    for j in range(pointsN):
      trajectory = Values[:, j, :]
      trajectory = [trajectory[..., i] for i in range(trajectory.shape[-1])]
      x = ax.plot(*trajectory, linestyle='dotted')
      colors.append(x[0].get_color())
    # draw the current position
    ax.plot(values[:, 0], values[:, 1], values[:, 2], 'ro', markersize=3)
    # draw the original colors if present
    if originalColors is not None:
      for j in range(pointsN):
        ax.plot(
          originalColors[j, 0], originalColors[j, 1], originalColors[j, 2],
          'o', markersize=2, color=colors[j]
        )
      pass

    if clip is not None:
      ax.set_xlim(*clip)
      ax.set_ylim(*clip)
      ax.set_zlim(*clip)
    return
  return F

class CInterpolantVisualization:
  def __init__(self, model):
    self._model = model
    return
  
  def _generateFilename(self, **kwargs):
    folder = os.path.abspath('tmp')
    os.makedirs(folder, exist_ok=True)
    # remove videos older than 1 hour
    now = time.time()
    MAX_AGE = 1 * 60 * 60 # 1 hour
    for filename in glob.glob(os.path.join(folder, '*.mp4')):
      if MAX_AGE < now - os.path.getmtime(filename):
        os.remove(filename)
      continue
    # hash the parameters
    parameters = dict(**kwargs)
    del parameters['images']
    parameters['raw'] = hasharr(kwargs['raw'])
    hash = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    return os.path.join(folder, f'{hash}.mp4')
  
  def _collectSteps(self, points, initialValues, kwargs):
    NSamples = len(points)
    watcher = CSamplerWatcher(
      steps=1000, # 1000 steps at most
      tracked=dict(
        value=(NSamples, 3),
        x0=(NSamples, 3),
        x1=(NSamples, 3),
      )
    )
    reverseArgs = kwargs.pop('reverseArgs', {})
    assert not('initialValues' in reverseArgs), 'Unexpected initialValues in reverseArgs'
    _ = self._model(
      **kwargs,
      pos=points,
      initialValues=initialValues,
      reverseArgs=dict(
        **reverseArgs,
        algorithmInterceptor=watcher.interceptor(),
      ),
    )
    N = watcher.iteration + 1
    def extract(name, duplicateFirst=True):
      data = watcher.tracked(name).numpy()
      if duplicateFirst:
        data = np.concatenate([data[:1], data], axis=0)
      data = data[:N]
      assert data.shape[:2] == (N, NSamples), f'Unexpected shape of "{name}" ({data.shape})'
      return data
    
    return CCollectedSteps(
      value=extract('value', duplicateFirst=False),
      x0=extract('x0'),
      x1=extract('x1'),
      totalSteps=N,
      totalPoints=NSamples,
    )
  
  def _generateVideo(self, writer, originalColors, collectedSteps):
    plotColorCurve2d = _plotColorCurve2d(collectedSteps, originalColors)
    plotEuclideanDistanceCurve2d = _plotEuclideanDistanceCurve2d(collectedSteps, originalColors)
    plotStepsTrajectories = _plotRGBTrajectories(
      Values=collectedSteps.value, originalColors=originalColors, title='Color trajectories (RGB)',
      clip=(-1.0, 1.0)
    )
    plotEstimatedTrajectories = _plotRGBTrajectories(
      Values=collectedSteps.x0, originalColors=originalColors,
      title='Estimated values trajectories (RGB)',
      clip=(-1.0, 1.0)
    )
    # contains the trajectory of color and also distance to the original color
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)
    axs = [
      fig.add_subplot(gs[0, 0]),
      fig.add_subplot(gs[1, 0]),
      fig.add_subplot(gs[0, 1], projection='3d'),
      fig.add_subplot(gs[1, 1], projection='3d'),
    ]
    try:
      for step in range(collectedSteps.totalSteps):
        fig.suptitle(f'Step {step+1}/{collectedSteps.totalSteps}')
        plotColorCurve2d(step=step, ax=axs[0])
        plotEuclideanDistanceCurve2d(step=step, ax=axs[1])
        plotStepsTrajectories(step=step, ax=axs[2])
        plotEstimatedTrajectories(step=step, ax=axs[3])

        fig.tight_layout()
        fig.show()
        writer.append_data(plot2image(fig))
        continue
    finally:
      plt.close(fig)
    return
  
  def _extractOriginalColors(self, points, raw, **kwargs):
    assert raw.ndim == 3, 'Unexpected shape of raw'
    assert raw.shape[-1] == 3, 'Unexpected number of channels in raw'
    assert raw.dtype == np.uint8, 'Unexpected dtype of raw'
    raw = raw.astype(np.float32) / 255.0
    res = extractInterpolated( ensure4d(raw[None]), points[None] ).numpy()[0]
    res = (res * 2.0) - 1.0 # [-1, 1]
    return res
  
  def _samplePoints(self, seed, numPoints, useOriginalColors, kwargs):
    points = np.random.RandomState(seed).uniform(size=(numPoints, 2)).astype(np.float32)
    originalColors = None
    if useOriginalColors:
      originalColors = self._extractOriginalColors(points=points, **kwargs)
    return points, originalColors

  def _initialValuesFor(self, initialValues, seed, N):
    if initialValues == 'Zeros': return np.zeros((1, N, 3), dtype=np.float32)
    if initialValues == 'Seeded':
      return np.random.RandomState(seed).uniform(size=(1, N, 3)).astype(np.float32)

    raise ValueError(f'Unexpected initialValues: {initialValues}')
  
  def __call__(self,
    VT_numPoints=None, VT_fps=None, VT_useOriginalColors=None, VT_seed=-1,
    VT_initialValues=None,
    **kwargs
  ):
    if VT_numPoints is None: return self._model(**kwargs)
    VT_numPoints = int(VT_numPoints)
    VT_seed = None if VT_seed < 0 else int(VT_seed)
    VT_fps = int(VT_fps)
    VT_useOriginalColors = bool(VT_useOriginalColors)

    videoFileName = self._generateFilename(
      VT_numPoints=VT_numPoints, VT_seed=VT_seed, VT_fps=VT_fps,
      VT_initialValues=VT_initialValues,
      VT_useOriginalColors=VT_useOriginalColors,
      **kwargs
    )
    # if os.path.exists(videoFileName): return {'video': videoFileName} # cache
    points, originalColors = self._samplePoints(
      seed=VT_seed, numPoints=VT_numPoints, useOriginalColors=VT_useOriginalColors,
      kwargs=kwargs,
    )
    
    with imageio.get_writer(videoFileName, mode='I', fps=VT_fps) as writer:
      self._generateVideo(
        writer=writer,
        originalColors=originalColors,
        collectedSteps=self._collectSteps(
          points=points,
          initialValues=self._initialValuesFor(
            initialValues=VT_initialValues, seed=VT_seed, N=VT_numPoints
          ),
          kwargs=kwargs,
        ),
      )
    return { 'video': videoFileName, }
  
  @property
  def kind(self): return self._model.kind
  
  @property
  def name(self): return self._model.name
# End of CInterpolantVisualization