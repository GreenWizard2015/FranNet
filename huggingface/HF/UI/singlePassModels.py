import gradio as gr
import os
from .common import modelNameSelector, bindClick, markdownFrom

def singlePassModels(processImage, models, commonSettings, resultActions):
  with gr.Tab(label='Single-pass models'):
    with gr.Row():
      with gr.Column():
        markdownFrom(os.path.join(os.path.dirname(__file__), 'markdown', 'single-pass-about.md'))
        modelName = modelNameSelector(models, kind='single-pass')
        submit = gr.Button(value='Submit')

      with gr.Column():
        upscaledImage = gr.Image(type='pil', label='Upscaled and colorized image', interactive=False)
        resultActions(upscaledImage)
        
      bindClick(
        submit, processImage, 
        dict(modelName=modelName, **commonSettings),
        dict(upscaled=upscaledImage)
      )
  return

