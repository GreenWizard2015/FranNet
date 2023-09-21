import gradio as gr
from .common import modelNameSelector, markdownFrom

def singlePassModels(models, submit):
  with gr.Tab(label='Single-pass models'):
    markdownFrom('single-pass-about.md')
    modelName = modelNameSelector(models, kind='single pass')
    submit(modelName=modelName)
  return

