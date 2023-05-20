import argparse, os, sys
# add the root folder of the project to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.utils import setupGPU, load_config
setupGPU() # call it on startup to prevent OOM errors on my machine

import tensorflow as tf
from NN.utils import ensure4d
from Utils import dataset_from_config
from NN.CBaseModel import CBaseModel

class COracleModel(CBaseModel):
  def __init__(self, grayscaled, method, **kwargs):
    super().__init__(**kwargs)
    self._grayscaled = grayscaled
    self._method = method
    return
  
  def test_step(self, images):
    (src, dest) = images
    src = ensure4d(src)
    dest = ensure4d(dest)
    srcShape = (tf.shape(src)[1], tf.shape(src)[2])
    destShape = (tf.shape(dest)[1], tf.shape(dest)[2])
    
    # create fake reconstructed image from the ground truth
    reconstructed = self.convertRange(dest, targetRange='0..1')
    # downscale it to the size of the source image and upscale back
    reconstructed = tf.image.resize(reconstructed, srcShape, method=self._method)
    if self._grayscaled:
      reconstructed = tf.image.rgb_to_grayscale(reconstructed)
      
    reconstructed = tf.image.resize(reconstructed, destShape, method=self._method)
    if self._grayscaled:
      reconstructed = tf.image.grayscale_to_rgb(reconstructed)
    # convert back to the range of the source image
    reconstructed = self.convertRange(reconstructed, targetRange='-1..1')

    return self._testMetrics(dest, reconstructed)
  
def main(args):
  folder = os.path.dirname(__file__)
  config = load_config(args.config, folder=folder)
  # Select dataset
  dataset = dataset_from_config(config['dataset'])
  test_data = dataset.make_dataset(config['dataset']['test'], split='test')
  
  measurements = {}
  for method in ['nearest', 'bilinear', 'bicubic', 'area']:
    print(f'Interpolation method: {method}')
    print()
    measurements[method] = {}
    methodMeasurements = measurements[method]
    for grayscaled in [True, False]:
      if grayscaled:
        print('Oracle that did not know real colors')
      else:
        print('Oracle that knew real colors')

      model = COracleModel(grayscaled=grayscaled, method=method)
      model.compile()
      losses = model.evaluate(test_data, return_dict=True, verbose=1)
      methodMeasurements[grayscaled] = {'RGB': losses['loss'], 'Grayscale': losses['loss_gr']}
      print('-' * 80)
      continue
    print('=' * 80)
    print()
    continue
  
  # Generate HTML table
  html_table = """<table class="myTable">
    <tr>
      <th></th>
      <th colspan="2">with color reference</th>
      <th colspan="2">without color reference</th>
    </tr>
    <tr>
      <th>Method</th>
      <th>RGB MSE</th>
      <th>Grayscale MSE</th>
      <th>RGB MSE</th>
      <th>Grayscale MSE</th>
    </tr>
  """

  for method, methodMeasurements in measurements.items():
    html_table += "<tr>\n" if not (method == 'bilinear') else "<tr class='used'>\n"
    html_table += f"  <td>{method}</td>\n"
    for grayscaled in [False, True]:
      losses = methodMeasurements[grayscaled]
      html_table += "  <td>%.05f</td>  <td>%.05f</td>\n" % (losses['RGB'], losses['Grayscale'])
    html_table += "</tr>\n"

  html_table += "</table>"

  print(html_table)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process arguments.')
  parser.add_argument(
    '--config', type=str, required=True,
    help='Path to a single config file or a multiple config files (they will be merged in order of appearance)',
    default=[], action='append', 
  )

  args = parser.parse_args()
  main(args)