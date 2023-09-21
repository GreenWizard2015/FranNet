import gradio as gr
from HF.UI.common import modelNameSelector, markdownFrom
from HF.UI.areas.DDIMParametersArea import DDIMParametersArea

def diffusionModels(models, submit):
  with gr.Tab(label='Diffusion models'):
    markdownFrom('diffusion-about.md')
    modelName = modelNameSelector(models, kind='diffusion')
    DDIMParameters = DDIMParametersArea()
    submit(
      modelName=modelName,
      **DDIMParameters,
    )
  return
