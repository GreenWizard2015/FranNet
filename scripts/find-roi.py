# this script takes a folder with images
# searches for the most interesting/distinctive regions in the images
# and draws the regions on the images
import argparse, os, sys, glob, cv2, numpy as np
# add the root folder of the project to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.visualize import withText, withPadding

def isImage(filename):
  filename = os.path.basename(filename).lower()
  if not filename.endswith(('.png', '.jpg')): return False
  return True

def regionsFor(H, W, regionSize):
  # create a grid of regions
  regions = []
  N = 8
  fractions = np.linspace(0, regionSize, N + 1, dtype=np.int32)[:-1]
  x, y = np.meshgrid(fractions, fractions)
  shifts = np.stack([y, x], axis=-1).reshape(-1, 2)
  shifts = map(tuple, shifts)
  shifts = list(set(shifts))

  for shiftY, shiftX in shifts:
    for y in range(shiftY, H, regionSize):
      for x in range(shiftX, W, regionSize):
        if H < y + regionSize: continue
        if W < x + regionSize: continue
        regions.append((y, x, y + regionSize, x + regionSize))
        continue
      continue
  return list(set(regions))

def calculateStatistics(images, regions):
  assert 1 < len(images), 'Need at least 2 images'
  H, W = images[0].shape[:2]
  candidates = []
  for region in regions:
    y1, x1, y2, x2 = region
    # verify that the region is inside the image
    assert 0 <= y1 < y2 <= H, f'{y1} {y2} {H}'
    assert 0 <= x1 < x2 <= W, f'{x1} {x2} {W}'

    # extract the region from each image
    regionImages = [image[y1:y2, x1:x2] for image in images]
    globalStd = max([np.std(x) for x in regionImages])
    # (H, W, C) * N -> (H, W, C, N)
    regionImages = np.stack(regionImages, axis=-1)
    assert regionImages.shape[-1] == len(images), 'Wrong number of images'
    localStd = np.std(regionImages, axis=-1).max(axis=-1).mean()
    differences = globalStd + localStd
    assert isinstance(differences, float), 'Expected a scalar'
    candidates.append((differences, region))
    continue
  return candidates

def region2points(region):
  y1, x1, y2, x2 = region
  assert x1 < x2, 'Wrong region'
  assert y1 < y2, 'Wrong region'
  A = (x1, y1)
  B = (x2, y1)
  C = (x2, y2)
  D = (x1, y2)
  return A, B, C, D

def isOverlapping(region1, region2):
  y1, x1, y2, x2 = region1
  assert x1 < x2, 'Wrong region'
  assert y1 < y2, 'Wrong region'
  corners = region2points(region1)
  for point in region2points(region2):
    x, y = point
    if (x1 <= x <= x2) and (y1 <= y <= y2) and not(point in corners):
      return True
    continue
  return False

def hasCommonEdge(region1, region2):
  pts1 = region2points(region1)
  pts2 = region2points(region2)
  commonPointsN = sum([int(pt in pts2) for pt in pts1])
  return commonPointsN == 2

def drawRegions(images, regions, mergeCommonEdges=True):
  images = images if isinstance(images, list) else [images]
  if mergeCommonEdges:
    wasMerged = True
    while wasMerged:
      wasMerged = False
      for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
          if i == j: continue
          if hasCommonEdge(region1, region2):
            # replace the regions with a merged region
            y1 = min(region1[0], region2[0])
            x1 = min(region1[1], region2[1])
            y2 = max(region1[2], region2[2])
            x2 = max(region1[3], region2[3])
            regions[i] = (y1, x1, y2, x2)
            regions.pop(j)
            wasMerged = True
            break
          continue
        if wasMerged: break
        continue
      continue
  # draw the regions on each subimage
  for region in regions:
    y1, x1, y2, x2 = region
    for image in images:
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
      continue
    continue
  return

def findBestRegions(regions, N):
  candidates = sorted(regions, key=lambda x: x[0], reverse=True)
  candidatesBest = []
  while len(candidatesBest) < N:
    if not candidates: break
    bestCandidate = candidates.pop(0)
    _, bestRegion = bestCandidate
    # check if the best candidate is not overlapping with the previous candidates
    overlapping = any(
      isOverlapping(bestRegion, candidateRegion)
      for _, candidateRegion in candidatesBest
    )
    if not overlapping:
      candidatesBest.append(bestCandidate)
    continue
  return candidatesBest[:N]

def processImages(images, regionSize, N):
  mainImage = images[0]
  H, W = mainImage.shape[:2]
  regions = regionsFor(H, W, regionSize)
  candidates = findBestRegions(
    regions=calculateStatistics(images, regions),
    N=N
  )
  regions = [region for _, region in candidates]
  return regions

def main(args):
  folder = args.folder
  # find all images in the folder
  files = [f for f in glob.glob(os.path.join(folder, '*.*')) if isImage(f)]
  files.sort()
  # load images
  images = [cv2.imread(file) for file in files]
  regions = processImages(images, args.regionSize, args.N)
  
  # create a grid of regions
  perSource = [[] for _ in range(len(images))]
  targetSize = args.targetSize or args.regionSize
  for region in regions:
    y1, x1, y2, x2 = region
    subimages = [image[y1:y2, x1:x2] for image in images]
    subimages = [cv2.resize(image, (targetSize, targetSize)) for image in subimages]
    if args.showSubregions:
      subregions = processImages(subimages, args.subRegionSize, args.subRegionsN or args.N)
      drawRegions(subimages, subregions)
      pass
    subimages = [withPadding(image, 5) for image in subimages]
    for i, image in enumerate(subimages):
      perSource[i].append(image)
      continue
    continue
  
  perSource = [np.concatenate(images, axis=0) for images in perSource]
  # add captions, basename of the file without extension
  captions = [os.path.basename(file) for file in files]
  # remove the extension
  captions = [os.path.splitext(caption)[0] for caption in captions]
  # add the caption to each image
  perSource = [
    withText(image, caption, scale=0.5)
    for image, caption in zip(perSource, captions)
  ]
  # create a grid of images
  regionsImages = np.concatenate(perSource, axis=1)

  if args.output:
    cv2.imwrite(args.output, regionsImages)
  else:
    cv2.imshow('regions', regionsImages)
    cv2.waitKey(0)
  return

if '__main__' == __name__:
  parser = argparse.ArgumentParser(description='Create a grid of most interesting regions')
  parser.add_argument('--folder', help='folder with images')
  parser.add_argument('--regionSize', type=int, help='size of the region', default=384)
  parser.add_argument('--targetSize', type=int, help='size of the target region', default=256)
  parser.add_argument('--N', type=int, help='number of global regions', default=4)
  parser.add_argument('--hideSubregions', action='store_true', help='hide subregions')
  parser.add_argument('--subRegionsN', type=int, help='number of subregions', default=6)
  parser.add_argument('--subRegionSize', type=int, help='size of the subregion', default=int(256/6))
  parser.add_argument('--output', help='output file')

  args = parser.parse_args()
  args.showSubregions = not args.hideSubregions
  main(args)
  pass