from Utils.CImageProcessor import CImageProcessor

def CelebaImageProcessor(image_size, to_grayscale):
  return CImageProcessor(
    image_size=image_size,
    to_grayscale=to_grayscale,
    format='RGB', # in CelebA images are in RGB format
    range='0..255' # in CelebA images are in the 0..255 range
  )
