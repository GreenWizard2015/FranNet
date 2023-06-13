import gradio as gr
import os
from .common import modelNameSelector, markdownFrom
from .ARSamplerParameters import ARSamplerParameters

def _ARDirectionModels(models, submit):
  with gr.Tab(label='Direction'):
    markdownFrom(os.path.join(os.path.dirname(__file__), 'markdown', 'autoregressive-direction.md'))
    modelName = modelNameSelector(models, kind='autoregressive direction')
    #######################
    samplerParameters = ARSamplerParameters()
    #######################
    submit(
      modelName=modelName,
      **samplerParameters
    )
  return

def autoregressiveModels(models, submit):
  with gr.Tab(label='Autoregressive models'):
    markdownFrom(os.path.join(os.path.dirname(__file__), 'markdown', 'autoregressive-about.md'))
    # subtabs
    _ARDirectionModels(models, submit)
  return
