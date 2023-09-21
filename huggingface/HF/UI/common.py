import gradio as gr
import os

def modelNameSelector(models, kind):
  models = { k: v for k, v in models.items() if kind == v.kind }
  names = sorted(models.keys())
  frst = names[0] if 0 < len(names) else None
  modelName = gr.Dropdown(label='Model name', choices=names, value=frst, interactive=True, allow_custom_value=False)
  return modelName

# helper to bind click event to a function with named inputs (not implemented in gradio)
def bindClick(btn, fn, inputs, outputs):
  assert isinstance(inputs, dict), 'inputs must be a dict'
  assert isinstance(outputs, dict), 'outputs must be a dict'
  inputs = list(inputs.items())
  outputs = list(outputs.items())
  inputsNames = [ name for name, _ in inputs ]
  outputsNames = [ name for name, _ in outputs ]

  def handler(*args):
    assert len(args) == len(inputsNames), 'invalid number of arguments'
    # convert inputs list to dict with corresponding keys
    inputsDict = {name: value for name, value in zip(inputsNames, args)}
    res = fn(**inputsDict)
    assert isinstance(res, dict), 'function must return a dict'
    diff = set(res.keys()) - set(outputsNames)
    assert 0 == len(diff), f'invalid output names: {diff}'
    # convert outputs dict to list with corresponding values
    res = [res[name] for name in outputsNames]
    if 1 == len(res): res = res[0]
    return res

  btn.click(
    handler, 
    inputs=[v for _, v in inputs],
    outputs=[v for _, v in outputs]
  )
  return

# read markdown file and create a markdown widget
def markdownFrom(*path):
  folder = os.path.dirname(__file__)
  path = os.path.join(folder, 'markdown', *path)
  try:
    with open(path, 'r') as f:
      text = f.read()
  except FileNotFoundError:
    text = f'File not found: {path}'
  return gr.Markdown(text)

def noiseProviderStddev(value=None):
  if value is None: value = 'normal'
  return gr.Radio(
    ['normal', 'squared', 'zero'], label='Noise stddev', interactive=True,
    value=value
  )