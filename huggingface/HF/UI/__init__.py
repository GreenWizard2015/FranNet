import gradio as gr
from .areas.commonSettings import commonSettings
from .singlePassModels import singlePassModels
from .diffusionModels import diffusionModels
from .autoregressiveModels import autoregressiveModels
from .common import markdownFrom, bindClick
from .areas.ablationArea import ablationArea
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

def submitsCollector():
  submitsAreas = []

  def onSubmit(btn=None, **kwargs):
    submit = gr.Button(value='Submit') if btn is None else btn
    submitsAreas.append((submit, kwargs))
    return submit
  
  def onFinish(settings, upscaledImage, handler):
    for submit, kwargs in submitsAreas:
      outputs = kwargs.pop('outputs', {})
      bindClick(
        submit, handler,
        inputs=dict(kwargs, **settings),
        outputs={'upscaled': upscaledImage, **outputs}
      )
      continue
    return
  
  return onSubmit, onFinish

def AppUI(preprocessImage, processImage, models):
  resultActions, onFinishResults = resultsCollector()
  submitAction, onFinishSubmits = submitsCollector()
  with gr.Blocks() as app:
    markdownFrom('about.md')
    # common settings for all models
    settings = commonSettings(preprocessImage, resultActions)
    # ablation study area
    ablationFlags = ablationArea()
    # tabs for each model kind
    gr.Markdown('# Models')
    with gr.Row():
      with gr.Column():
        modelsKinds = [ singlePassModels, diffusionModels, autoregressiveModels ]
        for tabFor in modelsKinds:
          tabFor(models, submit=submitAction)
          continue
      
      with gr.Column():
        upscaledImage = gr.Image(type='pil', label='Upscaled and colorized image', interactive=False)
        resultActions(upscaledImage)

    # results comparison
    # TODO: show from which model each image is
    gr.Markdown('# Results comparison')
    with gr.Row():
      with gr.Column():
        leftImage = gr.Image(type='pil', show_label=False, label='Left image')

      with gr.Column():
        rightImage = gr.Image(type='pil', show_label=False, label='Right image')

      onFinishResults(leftImage, rightImage)
      onFinishSubmits(
        dict(**settings, **ablationFlags),
        upscaledImage, processImage
      )
      pass
  return app