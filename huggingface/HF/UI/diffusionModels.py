import gradio as gr
import os
from .common import modelNameSelector, markdownFrom, noiseProviderStddev

def _presets():
  # (stochasticity, K, clipping, projectNoise, noiseStddev)
  presets = {
    'Best': (1.0, 8, True, True, 'normal'),
    'DDPM': (1.0, 1, False, False, 'normal'),
    'DDIM': (0.0, 1, False, False, 'normal'),
  }
  def onSelect(presetName):
    preset = presets.get(presetName, None)
    assert preset is not None, f'Unknown preset: {presetName}'
    return preset
  
  names = list(presets.keys())
  defaultPreset = 'Best'
  defaultPresetValues = presets[defaultPreset]
  return names, onSelect, {
    'name': defaultPreset,
    'stochasticity': defaultPresetValues[0],
    'K': defaultPresetValues[1],
    'clipping': defaultPresetValues[2],
    'projectNoise': defaultPresetValues[3],
    'noiseStddev': defaultPresetValues[4],
  }

def diffusionModels(models, submit):
  presets, onSelectPreset, defaultPreset = _presets()
  with gr.Tab(label='Diffusion models'):
    markdownFrom(os.path.join(os.path.dirname(__file__), 'markdown', 'diffusion-about.md'))
    modelName = modelNameSelector(models, kind='diffusion')
    #######################
    with gr.Group():
      preset = gr.Dropdown(
        choices=presets, value=presets[0], label='Parameters preset',
        interactive=True, allow_custom_value=False
      )
      # DDIM parameters
      stochasticity = gr.Slider(
        minimum=0.0, maximum=1.0, step=0.05, label='Stochasticity', interactive=True,
        value=defaultPreset['stochasticity']
      )
      K = gr.Slider(
        minimum=1, maximum=30, step=1, label='K', interactive=True,
        value=defaultPreset['K']
      )
      clipping = gr.Checkbox(label='Clipping to [-1, 1]', interactive=True, value=defaultPreset['clipping'])
      projectNoise = gr.Checkbox(label='Use noise projection', interactive=True, value=defaultPreset['projectNoise'])
      noiseStddev = noiseProviderStddev(defaultPreset['noiseStddev'])

      # bind preset to parameters
      preset.change(
        onSelectPreset,
        inputs=[preset],
        outputs=[stochasticity, K, clipping, projectNoise, noiseStddev],
      )
    #######################
    submit(
      modelName=modelName,
      # DDIM parameters
      stochasticity=stochasticity,
      K=K,
      clipping=clipping,
      projectNoise=projectNoise,
      noiseStddev=noiseStddev
    )
  return
