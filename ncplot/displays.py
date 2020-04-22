#======================================================================================================================#
#================================================== DISPLAY 1D ========================================================#
#======================================================================================================================#
def display_1D(data):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(data)
    plt.xlim(0, len(data))
    plt.xlabel('Time')
    plt.ylabel(data.name + ' (' + data.units + ')')
    plt.show()
#======================================================================================================================#
#================================================== DISPLAY 2D ========================================================#
#======================================================================================================================#


def display_2D(data):
    #TODO:ajouter les slickers sur l'altitude et le temps
    import plotly.graph_objects as go
    from numpy import linspace, arange

    # fig = go.Figure(data = go.Contour(z=[data[0,0,:,:], data[1,0,:,:]]))
    #    fig.update_layout(
    #        xaxis=dict(
    #            tickmode='array',
    #            tickvals=linspace(0,len(data[0,0,0,:]), num=10, endpoint=True),
    #            ticktext=linspace(0, 360, num=10, endpoint=True)
    #        ),
    #        yaxis=dict(
    #            tickmode='array',
    #            tickvals=linspace(0,len(data[0,0,:,0]), num=10, endpoint=True),
    #            ticktext=linspace(-90, 90, num=10, endpoint=True)
    #        ),
    #        title = "Title",
    #        xaxis_title = "Longitude (°E)",
    #        yaxis_title = "Latitude (°N)",
    #    )
    #fig.show()

    # Build all traces with visible=False
    traces = [go.Contour(z=data[0,i,:,:]) for i in arange(0, len(data[0,:,0,0]))]


    # Build slider steps
    steps = []
    for i in range(len(traces)):
        step = dict(
            # Update method allows us to update both trace and layout properties
            method='update',
            args=[
                # Make the ith trace visible
                {'visible': [t == i for t in range(len(data))]},]

                # Set the title for the ith trace
                #{'title.text': 'Time (s) %d' % i}],
        )
        steps.append(step)

    # Build sliders
    sliders = [go.layout.Slider(
        active=10,
        currentvalue={},
        pad={},
        steps=steps
    )]

    layout = go.Layout(
        sliders=sliders,
            xaxis=dict(
                tickmode='array',
                tickvals=linspace(0,len(data[0,0,0,:]), num=10, endpoint=True),
                ticktext=linspace(0, 360, num=10, endpoint=True)
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=linspace(0,len(data[0,0,:,0]), num=10, endpoint=True),
                ticktext=linspace(-90, 90, num=10, endpoint=True)
            ),
            title = "Title",
            xaxis_title = "Longitude (°E)",
            yaxis_title = "Latitude (°N)",
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()
#======================================================================================================================#
#================================================== DISPLAY 3D ========================================================#
#======================================================================================================================#


def display_3D(data):
    import plotly.plotly as py
    from plotly.grid_objs import Grid, Column

    import time
    import numpy as np


    my_columns = []
    nr_frames = 68
    for k in range(nr_frames):
        my_columns.extend(
            [Column((6.7 - k * 0.1) * np.ones((r, c)), 'z{}'.format(k + 1)),
             Column(np.flipud(volume[67 - k]), 'surfc{}'.format(k + 1))]
        )
    grid = Grid(my_columns)
    py.grid_ops.upload(grid, 'anim_sliceshead' + str(time.time()), auto_open=False)

    data = [
        dict(
            type='surface',
            zsrc=grid.get_column_reference('z1'),
            surfacecolorsrc=grid.get_column_reference('surfc1'),
            colorscale=pl_bone,
            colorbar=dict(thickness=20, ticklen=4)
        )
    ]

    frames = []
    for k in range(nr_frames):
        frames.append(
            dict(
                data=[dict(zsrc=grid.get_column_reference('z{}'.format(k + 1)),
                           surfacecolorsrc=grid.get_column_reference('surfc{}'.format(k + 1)))],
                name='frame{}'.format(k + 1)
            )
        )

    sliders = [
        dict(
            steps=[dict(method='animate',
                        args=[['frame{}'.format(k + 1)],
                              dict(mode='immediate',
                                   frame=dict(duration=70, redraw=False),
                                   transition=dict(duration=0))],
                        label='{:d}'.format(k + 1)) for k in range(68)],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12),
                              prefix='slice: ',
                              visible=True,
                              xanchor='center'
                              ),
            len=1.0
        )
    ]

    axis3d = dict(
        showbackground=True,
        backgroundcolor="rgb(230, 230,230)",
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
    )

    layout3d = dict(
        title='Slices in volumetric data',
        font=dict(family='Balto'),
        width=600,
        height=600,
        scene=dict(xaxis=(axis3d),
                   yaxis=(axis3d),
                   zaxis=dict(axis3d, **dict(range=[-0.1, 6.8], autorange=False)),
                   aspectratio=dict(x=1, y=1, z=1),
                   ),
        updatemenus=[
            dict(type='buttons',
                 showactive=False,
                 y=1,
                 x=1.3,
                 xanchor='right',
                 yanchor='top',
                 pad=dict(t=0, r=10),
                 buttons=[dict(label='Play',
                               method='animate',
                               args=[
                                   None,
                                   dict(frame=dict(duration=70, redraw=False),
                                        transition=dict(duration=0),
                                        fromcurrent=True,
                                        mode='immediate')
                               ])])
        ],
        sliders=sliders
    )

    fig = dict(data=data, layout=layout3d, frames=frames)
    py.icreate_animations(fig, filename='animslicesHead' + str(time.time()))

