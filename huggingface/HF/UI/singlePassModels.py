import gradio as gr
import os
from .common import modelNameSelector, markdownFrom

def singlePassModels(models, submit):
  with gr.Tab(label='Single-pass models'):
    markdownFrom(os.path.join(os.path.dirname(__file__), 'markdown', 'single-pass-about.md'))
    modelName = modelNameSelector(models, kind='single pass')
    submit(modelName=modelName)
  return

