import argparse, os, sys, glob, cv2, numpy as np
# add the root folder of the project to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.visualize import withText, makeGrid

def imagePreprocessor(args):
  P = args.padding
  def f(filename):
    img = cv2.imread(filename)
    if img is None: return None
    if args.size is not None:
      img = cv2.resize(img, (args.size, args.size))

    if args.padding is not None:
      img = np.pad(img, ((P, P), (P, P), (0, 0)), mode='constant', constant_values=255)
    return img
  return f

def process(folder, args):
  # collect images (png or jpg), sort them by name
  files = [
    f for f in glob.glob(os.path.join(folder, '*.*')) 
    if f.lower().endswith(('.png', '.jpg')) and not f.lower().endswith(('grid.jpg', 'panorama.jpg'))
  ]
  files.sort()
  # load images
  preprocessor = imagePreprocessor(args)
  images = [preprocessor(file) for file in files]
  images = [img for img in images if img is not None] # remove any failed images
  if len(images) == 0:
    print(f'No images found in "{folder}"')
    return
  # create grid with 'columns' or less columns
  grid = makeGrid(images, columns=min(args.columns, len(images)))

  # add text, if provided
  if args.text is not None:
    folderName = os.path.basename(folder)
    # remove leading numbers and spaces
    folderName = folderName.lstrip('0123456789 ').lstrip()
    text = args.text.format(folder=folderName)
    grid = withText(grid, position='top', text=text, scale=0.5)
  return grid

def main(args):
  folder = args.folder
  # collect subfolders recursively, including the root folder, using glob
  subfolders = [folder]
  for subfolder in glob.glob(os.path.join(folder, '**'), recursive=True):
    if os.path.isdir(subfolder):
      subfolders.append(subfolder)
    continue
  subfolders = list(set(subfolders)) # remove duplicates, just in case
  subfolders.sort() # sort alphabetically
  # process each subfolder
  panorama = []
  for subfolder in subfolders:
    print(f'Processing "{subfolder}"...')
    img = process(subfolder, args)
    if img is None: continue
    if args.panorama:
      panorama.append(img)
    else:
      cv2.imwrite(os.path.join(subfolder, 'grid.jpg'), img)
    continue

  if args.panorama:
    panorama = makeGrid(panorama, columns=min(args.panoramaColumns or args.columns, len(panorama)))
    cv2.imwrite(os.path.join(folder, 'panorama.jpg'), panorama)
  return

if '__main__' == __name__:
  parser = argparse.ArgumentParser(description='Generate grid from folder')
  # parser.add_argument('--folder', help='folder with images or subfolders with images')
  # parser.add_argument('--columns', type=int, default=5, help='number of columns')
  # parser.add_argument('--padding', type=int, default=5, help='padding between images')
  # parser.add_argument('--size', type=int, default=256, help='size of the images')
  # parser.add_argument('--text', type=str, help='text to display at the top of the grid (optional)')
  # parser.add_argument('--panorama', action='store_true', help='create a panorama instead of a grid')
  # parser.add_argument('--panoramaColumns', type=int, default=50, help='number of columns in the panorama')

  args = parser.parse_args()
  # used for creating an illustrations
  args.folder = 'd:/visualized/2/'
  args.columns = 1
  args.padding = 5
  args.size = 256
  args.text = '{folder}'
  args.panorama = True
  args.panoramaColumns = 5
  main(args)
  pass