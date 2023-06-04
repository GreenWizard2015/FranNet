# script to run the app in HuggingFace space
import os, argparse
import gradio as gr
import numpy as np
from PIL import Image

from Utils.CImageProcessor import CImageProcessor

def toPILImage(img, isBGR=False):
  if not (np.uint8 == img.dtype): img = (img * 255).astype(np.uint8)
  if 2 == len(img.shape): img = img[..., None]
  if 1 == img.shape[-1]: img = np.repeat(img, 3, axis=-1) # grayscale
  if isBGR: img = img[..., ::-1]

  assert 3 == len(img.shape), f'Invalid image shape: {img.shape}'
  assert np.uint8 == img.dtype, f'Invalid image dtype: {img.dtype}'
  assert 3 == img.shape[-1], f'Invalid image channels: {img.shape[-1]}'
  return Image.fromarray(img, 'RGB')

def main(args):
  folder = os.path.dirname(os.path.abspath(__file__))
  # Processor used for CelebA
  image_processor = CImageProcessor(
    image_size=64,
    reverse_channels=False,
    to_grayscale=True,
    normalize_range=True
  )
  def processImage(img):
    assert 3 == len(img.shape), f'Invalid image shape: {img.shape}'
    assert np.uint8 == img.dtype, f'Invalid image dtype: {img.dtype}'
    assert 3 == img.shape[-1], f'Invalid image channels: {img.shape[-1]}'

    input, _ = image_processor.process(img[None])
    input = image_processor.unnormalizeImg(input).numpy()
    upscaled = input
    return [
      toPILImage(input[0], isBGR=False),
      toPILImage(upscaled[0], isBGR=True)
    ]
  
  grFace = gr.Interface(
    fn=processImage, 
    inputs=[
      gr.Image(type='numpy', label='Input image'),
      # Target resolution, from 128 to 1024, default 512
      gr.Slider(minimum=128, maximum=1024, step=1, default=512, label='Target resolution'),
    ],
    outputs=[
      gr.Image(type='pil', label='Input to the model'),
      gr.Image(type='pil', label='Upscaled and colorized image')
    ],
    title='',
    description='',
  )
  
  grFace.launch(inline=False, server_port=args.port, server_name=args.host)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=7860)
  parser.add_argument('--host', type=str, default=None)
  args = parser.parse_args()
  main(args)