import argparse, os, sys, json
# add the root folder of the project to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../../'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # for 3d plots magic
from Utils.utils import JSONHelper
import cv2
from Utils.visualize import makeGrid
from Utils.CDDIMParameters import CDDIMParameters

def toSeries(data):
  res = list(data.items())

  # ensure that 'normal' is always last, 'zero' is always first
  stddevs = ['zero', 'squared', 'normal']
  isStddevs = any([d[0] in stddevs for d in res])
  if isStddevs:
    res.sort(key=lambda x: stddevs.index(x[0]) if x[0] in stddevs else len(stddevs))
  return res

def plot2D(data, plot={}):
  fig = plt.figure()
  for k, v in toSeries(data):
    x = [p['x'] for p in v]
    y = [p['y'] for p in v]
    args = plot
    if callable(args): args = args(k, v)
    plt.plot(x, y, label=k, **args)
    continue
  return fig

def plot3D(data, zlabel, plot={}):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  for k, points in toSeries(data):
    x = [p['x'] for p in points]
    y = [p['y'] for p in points]
    z = [p['z'] for p in points]
    args = plot
    if callable(args): args = args(k, points)
    ax.scatter(x, y, z, label=k, **args)
    continue

  ax.set_zlabel(zlabel)
  return fig

def plotData(data, xlabel, ylabel, zlabel=None, title=None, plot={}, beforeRender=None):
  if zlabel is None:
    plot2D(data, plot)
  else:
    plot3D(data, zlabel, plot)

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend()

  fig = plt.gcf()
  if beforeRender is not None: beforeRender(fig)
  
  fig.canvas.draw()

  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data[...,::-1] # convert RGB to BGR, default for cv2

def groupBy(data, key):
  groups = {}
  for d in data:
    k = d[key]
    if k not in groups: groups[k] = []
    groups[k].append(d)
    continue
  return groups

def toPoints(data, xKey, yKey, zKey=None):
  if isinstance(data, dict):
    return {
      k: toPoints(v, xKey, yKey, zKey)
      for k, v in data.items()
    }
  
  if zKey is not None:
    return [dict(x=d[xKey], y=d[yKey], z=d[zKey]) for d in data]
  
  return [dict(x=d[xKey], y=d[yKey]) for d in data]

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Create plots from collected data.')
  parser.add_argument('--input', type=str, help='Path to input file', default='docs/other/DDIM-data.json')
  parser.add_argument('--output', type=str, help='Path to output folder (optional)', default='docs/img/diffusion-restorator/')

  args = parser.parse_args()
  ###############################################
  data = JSONHelper.load(args.input)  
  # create plots
  def plot_All_basic():
    # skip type=uniform, no clipping, no noise projection
    imgWhole = plotData(
      data=toPoints(
        groupBy([
            d for d in data
            if (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False)
          ], 'noise stddev'
        ),
        xKey='stochasticity', yKey='K', zKey='loss'
      ),
      xlabel='stochasticity', ylabel='K', zlabel='loss',
      title='',
    )
    imgA = plotData(
      data=toPoints(
        groupBy([
            d for d in data
            if (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False)
            and (0.01 <= d['loss'])
          ], 'noise stddev'
        ),
        xKey='stochasticity', yKey='K', zKey='loss'
      ),
      xlabel='stochasticity', ylabel='K', zlabel='loss',
      title='0.01 <= loss',
    )
    imgB = plotData(
      data=toPoints(
        groupBy([
            d for d in data
            if (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False)
            and (0.007 <= d['loss'] < 0.01)
          ], 'noise stddev'
        ),
        xKey='stochasticity', yKey='K', zKey='loss'
      ),
      
      xlabel='stochasticity', ylabel='K', zlabel='loss',
      title='0.007 <= loss < 0.01',
    )

    CData = [
      d for d in data
      if (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False)
      and (d['loss'] <= 0.007)
    ]
    CBest = min(CData, key=lambda x: x['loss'])
    groups = groupBy(CData, 'noise stddev')
    groups['best'] = [CBest]
    imgC = plotData(
      data=toPoints(
        groups,
        xKey='stochasticity', yKey='K', zKey='loss'
      ),
      
      xlabel='stochasticity', ylabel='K', zlabel='loss',
      title='loss <= 0.007',
    )
    p = makeGrid([imgWhole, imgA, imgB, imgC], 2)
    cv2.imwrite(os.path.join(args.output, 'all-basic.png'), p)
    return
  
  def plot_K1_basic():
    # with K=1, skip type=uniform, no clipping, no noise projection
    imgA = plotData(
      data=toPoints(
        groupBy(
          [
            d for d in data
            if (d['K'] == 1) and (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False)
          ], 'noise stddev'
        ),
        xKey='stochasticity', yKey='loss'
      ),
      
      xlabel='stochasticity', ylabel='loss',
      title='K=1, skip type=uniform, no clipping, no noise projection',
    )
    # same, but without stddev=normal and 0.5<=stochasticity<=1.0
    imgB = plotData(
      data=toPoints(
        groupBy(
          [
            d for d in data
            if (d['K'] == 1) and (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False)
            and not (d['noise stddev'] == 'normal') and (d['stochasticity'] >= 0.5)
          ], 'noise stddev'
        ),
        xKey='stochasticity', yKey='loss'
      ),
      
      xlabel='stochasticity', ylabel='loss',
      title='Zoomed in',
    )

    combined = makeGrid([imgA, imgB], 2)
    cv2.imwrite(os.path.join(args.output, 'K1-basic.png'), combined)
    return
  
  def plot_K1_clipping():
    ylims = [None]
    yline = [None]
    def onRenderA(fig):
      ax = fig.axes[0]
      ylims[0] = ax.get_ylim()
      # get min y value for stddev normal
      normLine = ax.get_lines()[-1]
      yline[0] = min(normLine.get_ydata())
      # draw dotted horizontal line
      ax.axhline(yline[0], linestyle='--', color='black')
      return
    
    def onRenderB(fig):
      fig.axes[0].set_ylim(ylims[0])
      fig.axes[0].axhline(yline[0], linestyle='--', color='black')
      return      

    imgA = plotData(
      data=toPoints(
        groupBy(
          [
            d for d in data
            if (d['K'] == 1) and (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False)
          ], 'noise stddev'
        ),
        xKey='stochasticity', yKey='loss'
      ),
      
      xlabel='stochasticity', ylabel='loss',
      title='K=1, skip type=uniform, no clipping, no noise projection',
      beforeRender=onRenderA,
    )
    dataA = [
      d for d in data
      if (d['steps skip type'] == 'uniform') and (d['clipping'] == True) and (d['project noise'] == False) and (d['K'] == 1)
    ]
    imgB = plotData(
      data=toPoints(
        groupBy(dataA, 'noise stddev'),
        xKey='stochasticity', yKey='loss'
      ),
      xlabel='stochasticity', ylabel='loss',
      title='skip type=uniform, K=1, with clipping',
      beforeRender=onRenderB,
    )
    ###################################
    # plot zoomed in versions
    dataNotclipped = [
      d for d in data
      if (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == False) and (d['K'] == 1)
      and (d['noise stddev'] != 'normal') and (d['stochasticity'] >= 0.5)
    ]
    dataClipped = [
      d for d in data
      if (d['steps skip type'] == 'uniform') and (d['clipping'] == True) and (d['project noise'] == False) and (d['K'] == 1)
      and (d['noise stddev'] != 'normal') and (d['stochasticity'] >= 0.5)
    ]

    groups = groupBy(dataNotclipped, 'noise stddev')
    groups.update({ k + ' clipped': v for k, v in groupBy(dataClipped, 'noise stddev').items() })
    def onPlot(name, values):
      # dotted if clipped, color as unclipped line
      if 'clipped' in name:
        name = name.replace(' clipped', '')
        # find unclipped line
        for line in plt.gcf().axes[0].get_lines():
          if line.get_label() == name:
            return {'linestyle': '--', 'color': line.get_color()}
          continue
        return {'linestyle': '--'}
      return {}
    
    imgC = plotData(
      data=toPoints(groups, xKey='stochasticity', yKey='loss'),
      xlabel='stochasticity', ylabel='loss',
      title='Zoomed in',
      plot=onPlot,
    )

    res = makeGrid([imgA, imgB, imgC], 3)
    cv2.imwrite(os.path.join(args.output, 'all-clipping.png'), res)
    return
  
  def plot_All_project_noise():
    PNData = [
      d for d in data
      if (d['steps skip type'] == 'uniform') and (d['clipping'] == False) and (d['project noise'] == True)
      and (d['K'] < 11)
    ]
    # skip type=uniform, no clipping, project noise
    stoch = [
      plotData(
        data=toPoints(
          groupBy([
              d for d in PNData
              if (d['stochasticity'] == stochasticity)
            ], 'noise stddev'
          ),
          xKey='K', yKey='loss'
        ),
        
        xlabel='K', ylabel='loss',
        title='noise projection, stochasticity=%0.1f' % (stochasticity,),
      )
      for stochasticity in [0.0, 0.5, 1.0]
    ]
    imgB = plotData(
      data=toPoints(
        groupBy(PNData, 'noise stddev'),
        xKey='stochasticity', yKey='K', zKey='loss'
      ),
      
      xlabel='stochasticity', ylabel='K', zlabel='loss',
      title='skip type=uniform, no clipping, project noise',
    )

    p = makeGrid([imgB] + stoch, 2)
    cv2.imwrite(os.path.join(args.output, 'all-project-noise.png'), p)
    return
  
  plot_All_basic()
  plot_K1_basic()
  plot_K1_clipping()
  plot_All_project_noise()
  ########################################
  # some further analysis, not used/finished
  
  # top 10 parameters
  best = sorted(data, key=lambda d: d['loss'])[:10]
  best = [CDDIMParameters.fromStats(d).toModelConfig() for d in best]
  JSONHelper.save('best-ddim-configs.json', best)
  exit(0)
  ########################################
  from collections import defaultdict
  import itertools

  def findCommonParams(data, ignoreKeys=[]):
    params = defaultdict(int)
    for d in data:
      keys = sorted([k for k in d.keys() if ('loss' not in k) and (k not in ignoreKeys)])
      keysPermutations = (
        [ keys ] + # length N
        [[k] for k in keys] # length 1
      )
      # other lengths
      for i in range(2, len(keys)):
        keysPermutations += list(itertools.combinations(keys, i))
        continue
      keysPermutations = list(set([ tuple(sorted(k)) for k in keysPermutations ]))
      # add all permutations of keys
      for k in keysPermutations:
        asStr = ', '.join([ '%s=%s' % (k, d[k]) for k in k ])
        params[asStr] += 1
        continue
      continue

    params = [ (k, v) for k, v in params.items() if (1 < v) ] # ignore params that are not common
    if len(params) == 0: return None
    # split params key into individual keys/values
    params = [ (dict([ (k, v) for k, v in [p.split('=') for p in k.split(', ')] ]), v) for k, v in params ]
    # filter out params that are too common (v == len(data))
    while 0 < len(params):
      mostCommon = max(params, key=lambda kv: kv[1])
      if mostCommon[1] < len(data): break
      mostCommon = mostCommon[0]
      params = [
        (
          { k: v for k, v in P.items() if k not in mostCommon.keys() },
          V
        )
        for P, V in params if P != mostCommon
      ]
      # remove duplicates manually
      newParams = {}
      for P, V in params:
        asStr = ', '.join(sorted([ '%s=%s' % (k, v) for k, v in P.items() ]))
        newParams[asStr] = (P, V)
        continue
      params = list(newParams.values())
      continue
    if len(params) == 0: return None
    params = sorted(params, key=lambda kv: kv[1], reverse=True)
    return params[0][0] # best params
  
  bestRuns = [ d for d in data if d['loss'] <= 0.0065 ]
  assert len(set([ d['steps skip type'] for d in bestRuns ])) == 1, 'all runs should have same skip type'
  # convert values to strings
  bestRuns = [ { k: str(v) for k, v in d.items() } for d in bestRuns ]

  print(len(bestRuns), 'runs with loss < 0.0066')
  #
  splits = []
  def splitRuns(runs, parentId):
    if len(runs) <= 1: return
    mostCommon = findCommonParams(runs)
    if mostCommon is None: return
    # split runs by most common param
    print('splitting by %s' % (mostCommon,))
    commonRuns = []
    otherRuns = []
    for r in runs:
      isCommon = all([ r[k] == v for k, v in mostCommon.items() ])
      if isCommon: 
        commonRuns.append(r)
      else:
        otherRuns.append(r)
      continue

    commonId = len(splits)
    splits.append((mostCommon, len(commonRuns), len(otherRuns), parentId))
    # split runs further
    splitRuns(commonRuns, commonId)
    splitRuns(otherRuns, parentId)
    return
  
  splitRuns(bestRuns, -1)
  for i, (params, common, other, parent) in enumerate(splits):
    print(i, params, common, other, parent)
    continue
  pass