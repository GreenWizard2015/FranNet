import gradio as gr
from .common import noiseProviderStddev, markdownFrom
import os

def ARSamplerParameters():
  with gr.Box():
    markdownFrom(os.path.join(os.path.dirname(__file__), 'markdown', 'ar-sampler.md'))

    threshold = gr.Slider(
      minimum=0.0, maximum=0.01, step=0.0001, label='Threshold',
      interactive=True, value=0.001
    )
    start = gr.Slider(
      minimum=0.0, maximum=1.0, step=0.0001, label='Start',
      interactive=True, value=1.0
    )
    end = gr.Slider(
      minimum=0.0, maximum=1.0, step=0.0001, label='End',
      interactive=True, value=0.001
    )
    steps = gr.Slider(
      minimum=1, maximum=1000, step=1, label='Steps',
      interactive=True, value=100
    )
    decay = gr.Slider(
      minimum=0.0, maximum=1.0, step=0.01, label='Decay',
      interactive=True, value=0.9
    )
    noiseStddev = noiseProviderStddev('normal')

    convergeThreshold = gr.Slider(
      minimum=0.0, maximum=10.0, step=0.0001, label='Converge threshold',
      interactive=True, value=0.0
    )
  return dict(
    threshold=threshold, convergeThreshold=convergeThreshold,
    start=start, end=end, steps=steps, decay=decay, # parameters for autoregressive direction sampler
    noiseStddev=noiseStddev
  )