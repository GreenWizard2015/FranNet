import gradio as gr
from .common import modelNameSelector, markdownFrom
from .areas.ARSamplerParameters import ARSamplerParameters
from .areas.visualizeTrajectoriesArea import visualizeTrajectoriesArea
from .areas.DDIMParametersArea import DDIMParametersArea

def _ARDirectionModels(models, submit):
  with gr.Tab(label='Direction'):
    markdownFrom('autoregressive-direction.md')
    modelName = modelNameSelector(models, kind='autoregressive direction')
    samplerParameters = ARSamplerParameters()
    #######################
    params = dict(modelName=modelName, **samplerParameters)
    visualizeTrajectoriesArea(
      lambda **extras: submit(**params, **extras),
    )
    submit(**params)
  return

def _ARDiffusionModels(models, submit):
  with gr.Tab(label='Diffusion'):
    # markdownFrom('autoregressive-diffusion.md')
    modelName = modelNameSelector(models, kind='autoregressive diffusion')
    DDIMParameters = DDIMParametersArea()
    #######################
    params = dict(modelName=modelName, **DDIMParameters)
    visualizeTrajectoriesArea(
      lambda **extras: submit(**params, **extras),
    )
    submit(**params)
  return

def autoregressiveModels(models, submit):
  with gr.Tab(label='Autoregressive models'):
    markdownFrom('autoregressive-about.md')
    # subtabs
    _ARDirectionModels(models, submit)
    _ARDiffusionModels(models, submit)
  return
