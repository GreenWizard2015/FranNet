import gradio as gr
import os, glob

def _examplesArea():
  folder = os.path.join(os.getcwd(), 'HF', 'examples')
  files = [img for img in glob.glob(f'{folder}/*.*') if img.lower().endswith(('png', 'jpg', 'jpeg'))]
  images = []
  with gr.Column():
    gr.Markdown('Examples')
    with gr.Row():
      for file in files:
        img = gr.Image(file, show_label=False, interactive=False)
        images.append(img)
        continue
      pass

  def onSelect(img):
    return img
  
  def bindToInputImage(inputImage):
    for img in images:
      img.select(onSelect, inputs=[img], outputs=[inputImage])
    return
  return bindToInputImage

def commonSettings(preprocessImage, resultActions):
  gr.Markdown('# Common settings')
  with gr.Row():
    with gr.Column():
      galleryConnector = _examplesArea()
      inputImage = gr.Image(type='numpy', label='Input image')
      resultActions(inputImage)
      galleryConnector(inputImage)
      targetResolution = gr.Slider(minimum=128, maximum=1024, step=1, value=512, label='Target resolution')
    
    with gr.Column():
      gr.Markdown('Actual input to the model')
      inputToModel = gr.Image(type='numpy', label='Input to the model', show_label=False, interactive=False)
      # TODO: find a way to prevent this image from being stretched

      # button "Replace input image"
      replaceInputImage = gr.Button(value='Replace input image')
      replaceInputImage.click(lambda x: x, inputs=[inputToModel], outputs=[inputImage])
    # Preprocess image on change
    inputImage.change(preprocessImage, inputs=[inputImage], outputs=[inputToModel])

  return {
    'inputImage': inputToModel,
    'targetResolution': targetResolution
  }
