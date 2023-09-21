import gradio as gr

def ablationArea():
  with gr.Accordion('Ablation area', open=False):
    randomizePositions = gr.Checkbox(label='Randomize positions for decoder', value=False)

    with gr.Accordion('Encoder', open=False):
      encoderContext = gr.Radio(
        label='Encoder contexts',
        choices=['Both', 'Only local', 'Only global'],
        value='Both',
      )

      intermediate = ['No intermediate %d' % i for i in range(1, 3 + 1)]
      intermediate.extend(['No intermediate %d (some models only)' % i for i in range(len(intermediate) + 1, 8)])
      intermediate = gr.CheckboxGroup(choices=intermediate, label='Intermediate contexts')
  
  return dict(
    randomizePositions=randomizePositions,
    encoderContext=encoderContext,
    encoderIntermediate=intermediate,
  )