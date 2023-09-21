import gradio as gr

def visualizeTrajectoriesArea(submit):
  with gr.Accordion('Visualize Trajectories Area', open=False):
    # number of points, button "Visualize", area for video
    numPoints = gr.Slider(
      label='Number of sampled points',
      value=10, minimum=1, maximum=200, step=1
    )
    fps = gr.Slider(label='Frames per second', value=2, minimum=1, maximum=30, step=1)
    VT_seed = gr.Number(
      label='Seed (-1 for random)', value=-1, 
      minimum=-1, maximum=1000000, step=1, interactive=True
    )
    VT_initialValues = gr.Radio(
      label='Initial values',
      choices=['Seeded', 'Zeros'], value='Seeded',
    )
    useOriginalColors = gr.Checkbox(label='Show original colors', value=False)
    button = gr.Button(value='Visualize')
    video = gr.Video(label='Video', interactive=False)
    # bind button
    submit(
      btn=button,
      VT_numPoints=numPoints, VT_fps=fps, VT_useOriginalColors=useOriginalColors,
      VT_initialValues=VT_initialValues,
      VT_seed=VT_seed,
      outputs={'video': video}
    )
    pass
  return