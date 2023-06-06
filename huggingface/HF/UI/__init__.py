import gradio as gr
from .commonSettings import commonSettings
from .singlePassModels import singlePassModels
from .diffusionModels import diffusionModels
from .common import markdownFrom
import os

def resultsCollector():
  resultsAreas = []

  def onResult(upscaledImage):
    with gr.Row():
      with gr.Column(): btnLeft = gr.Button(value='Send to left')
      with gr.Column(): btnRight = gr.Button(value='Send to right')
    resultsAreas.append((upscaledImage, btnLeft, btnRight))
    return
  
  def onFinish(leftImagePlaceholder, rightImagePlaceholder):
    for img, btnLeft, btnRight in resultsAreas:
      btnLeft.click(lambda x: x, inputs=[img], outputs=[leftImagePlaceholder])
      btnRight.click(lambda x: x, inputs=[img], outputs=[rightImagePlaceholder])
      continue
    return
  
  return onResult, onFinish

def AppUI(preprocessImage, processImage, models):
  resultActions, onFinish = resultsCollector()
  with gr.Blocks() as app:
    markdownFrom(os.path.join(os.path.dirname(__file__), 'markdown', 'about.md'))
    # common settings for all models
    settings = commonSettings(preprocessImage, resultActions)
    # tabs for each model kind
    gr.Markdown('# Models')
    modelsKinds = [ singlePassModels, diffusionModels ]
    for tabFor in modelsKinds:
      tabFor(processImage, models, commonSettings=settings, resultActions=resultActions)
      continue

    # results comparison
    # TODO: show from which model each image is
    gr.Markdown('# Results comparison')
    with gr.Row():
      with gr.Column():
        leftImage = gr.Image(type='pil', show_label=False, label='Left image')

      with gr.Column():
        rightImage = gr.Image(type='pil', show_label=False, label='Right image')
      onFinish(leftImage, rightImage)
      pass
  return app