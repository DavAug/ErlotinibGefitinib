#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pandas as pd
import pints
import plotly.colors
import plotly.graph_objects as go


def plot_measurements(data):
    """
    Returns a `plotly.graph_objects.Figure` containing a scatter plot of the
    tumour volume, as well as the mass time-series.

    This function assumes the follwing keys naming convention:
        ids: '#ID'
        time: 'TIME in day'
        tumour volume: 'TUMOUR VOLUME in cm^3
        mass: 'BODY WEIGHT in g'

    The axis labels as well as the hoverinfo assume that time is measured in
    day, volume is measured in cm^3, and mass is measured in g.

    Arguments:
        data -- A pandas.DataFrame containing the measured time-series data of
                the tumour volume and the mass.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            'Data has to be pandas.DataFrame.')

    keys = ['#ID', 'TIME in day', 'TUMOUR VOLUME in cm^3', 'BODY WEIGHT in g']
    for key in keys:
        if key not in data.keys():
            raise ValueError(
                'Data must have key <' + str(key) +
                '>.')

    # Get ids
    ids = data['#ID'].unique()

    # Get number of different ids
    n_ids = len(ids)

    # Define colorscheme
    colors = plotly.colors.qualitative.Plotly[:n_ids]

    # Create figure
    fig = go.Figure()

    # Scatter plot LXF A677 time-series data for each mouse
    for index, id_m in enumerate(ids):
        # Create mask for mouse
        mask = data['#ID'] == id_m

        # Get time points for mouse
        times = data['TIME in day'][mask]

        # Get observed tumour volumes for mouse
        observed_volumes = data['TUMOUR VOLUME in cm^3'][mask]

        # Get mass time series
        masses = data['BODY WEIGHT in g'][mask]

        # Plot tumour volume over time
        fig.add_trace(go.Scatter(
            x=times,
            y=observed_volumes,
            name="ID: %d" % id_m,
            showlegend=True,
            hovertemplate=(
                "<b>ID: %d</b><br>" % (id_m) +
                "Time: %{x:} day<br>"
                "Tumour volume: %{y:.02f} cm^3<br>"
                "Body weight: %{text}<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            text=['%.01f g' % mass for mass in masses],
            mode="markers",
            marker=dict(
                symbol='circle',
                color=colors[index],
                opacity=0.7,
                line=dict(color='black', width=1))
        ))

        # Plot mass over time
        fig.add_trace(go.Scatter(
            x=times,
            y=masses,
            name="ID: %d" % id_m,
            showlegend=True,
            visible=False,
            hovertemplate=(
                "<b>ID: %d</b><br>" % (id_m) +
                "Time: %{x:} day<br>"
                "Tumour volume: %{y:.02f} cm^3<br>"
                "Body weight: %{text}<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            text=['%.01f g' % mass for mass in masses],
            mode="markers",
            marker=dict(
                symbol='circle',
                color=colors[index],
                opacity=0.7,
                line=dict(color='black', width=1))
        ))

    # Set X, Y axis and figure size
    fig.update_layout(
        autosize=True,
        xaxis_title=r'$\text{Time in day}$',
        yaxis_title=r'$\text{Tumour volume in cm}^3$',
        template="plotly_white")

    # Add switch between linear and log y-scale
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"yaxis.type": "linear"}],
                        label="Linear y-scale",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.type": "log"}],
                        label="Log y-scale",
                        method="relayout"
                    )
                ]),
                pad={"r": 0, "t": -10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
            dict(
                type="buttons",
                direction="down",
                buttons=list([
                    dict(
                        args=[
                            {"visible": [True, False] * n_ids},
                            {"yaxis": {
                                "title": r'$\text{Tumour volume in cm}^3$'}}],
                        label="Tumour volume",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"visible": [False, True] * n_ids},
                            {"yaxis": {
                                "title": r'$\text{Body weight in g}$'}}],
                        label="Body weight",
                        method="update"
                    ),
                ]),
                pad={"r": 0, "t": -10},
                showactive=True,
                x=1.07,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Position legend
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=1.05))

    return fig


def plot_measurements_and_predictions(data, model, parameters):
    """
    Returns a `plotly.graph_objects.Figure` containing a scatter plot of the
    measured tumour volume time-series, and line plots of the model
    predictions.

    This function assumes the following keys naming convention:
        ids: '#ID'
        time: 'TIME in day'
        tumour volume: 'TUMOUR VOLUME in cm^3

    The axis labels as well as the hoverinfo assume that time is measured in
    day, volume is measured in cm^3.

    Arguments:
        data -- A pandas.DataFrame containing the measured time-series data of
                the tumour volume and the mass.
        model -- A `pints.ForwardModel`.
        parameters -- An array-like object with the model parameters for each
                      individual in the dataset.
                      Shape: (n_individuals, n_parameters)
    """
    # Check data has the correct type
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            'Data has to be pandas.DataFrame.')
    # Check that data has the required keys
    keys = ['#ID', 'TIME in day', 'TUMOUR VOLUME in cm^3', 'BODY WEIGHT in g']
    for key in keys:
        if key not in data.keys():
            raise ValueError(
                'Data must have key <' + str(key) +
                '>.')
    # Check that model has the correct type
    if not isinstance(model, pints.ForwardModel):
        raise TypeError(
            'Model needs to be an instance of `pints.ForwardModel`.')
    # Check that model has only one output dimension
    if model.n_outputs() != 1:
        raise ValueError(
            'Model output dimension has to be 1.')
    # Check that parameters have the correct dimensions
    parameters = np.asarray(parameters)
    if parameters.ndim != 2:
        raise ValueError(
            'Parameters needs to have dimension 2. Parameters has '
            'dimension <' + str(parameters.ndim) + '>.')

    # Get number of individuals
    n_ids = len(data['#ID'].unique())

    # Check that parameters have the correct shape
    if parameters.shape != (n_ids, model.n_parameters()):
        raise ValueError(
            'Parameters does not have the correct shape. Expected shape '
            '(n_individuals, n_parameters) = ' +
            str((n_ids, model.n_parameters())) + '.')

    # Define colorscheme
    colors = plotly.colors.qualitative.Plotly[:n_ids]

    # Create figure
    fig = go.Figure()

    # Scatter plot LXF A677 time-series data for each mouse
    ids = data['#ID'].unique()
    for index, id_m in enumerate(ids):
        # Create mask for mouse
        mask = data['#ID'] == id_m

        # Get observed data for indiviudal
        observed_times = data['TIME in day'][mask].to_numpy()
        observed_data = data['TUMOUR VOLUME in cm^3'][mask]

        # Simulate data
        params = parameters[index, :]
        start_experiment = data['TIME in day'].min()
        end_experiment = data['TIME in day'].max()
        simulated_times = np.linspace(
            start=start_experiment, stop=end_experiment)
        simulated_data = model.simulate(params, simulated_times)

        # Plot data
        fig.add_trace(go.Scatter(
            x=observed_times,
            y=observed_data,
            legendgroup="ID: %d" % id_m,
            name="ID: %d" % id_m,
            showlegend=True,
            hovertemplate=(
                "<b>Measurement </b><br>" +
                "ID: %d<br>" % id_m +
                "Time: %{x:} day<br>" +
                "Tumour volume: %{y:.02f} cm^3<br>" +
                "Cancer type: LXF A677<br>" +
                "<extra></extra>"),
            mode="markers",
            marker=dict(
                symbol='circle',
                opacity=0.7,
                line=dict(color='black', width=1),
                color=colors[index])
        ))

        # Plot simulated growth
        fig.add_trace(go.Scatter(
            x=simulated_times,
            y=simulated_data,
            legendgroup="ID: %d" % id_m,
            name="ID: %d" % id_m,
            showlegend=False,
            hovertemplate=(
                "<b>Simulation </b><br>" +
                "ID: %d<br>" % id_m +
                "Time: %{x:.0f} day<br>" +
                "Tumour volume: %{y:.02f} cm^3<br>" +
                "Cancer type: LXF A677<br>" +
                "<br>" +
                "<b>Parameter estimates </b><br>" +
                "Initial tumour volume: %.02f cm^3<br>" % params[0] +
                "Expon. growth rate: %.02f 1/day<br>" % params[1] +
                "Lin. growth rate: %.02f cm^3/day<br>" % params[2] +
                "<extra></extra>"),
            mode="lines",
            line=dict(color=colors[index])
        ))

    # Set X, Y axis and figure size
    fig.update_layout(
        autosize=True,
        xaxis_title=r'$\text{Time in day}$',
        yaxis_title=r'$\text{Tumour volume in cm}^3$',
        template="plotly_white")

    # Add switch between linear and log y-scale
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"yaxis.type": "linear"}],
                        label="Linear y-scale",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.type": "log"}],
                        label="Log y-scale",
                        method="relayout"
                    )
                ]),
                pad={"r": 0, "t": -10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )

    return fig


def plot_measurements_and_simulation(
        data, model, default_parameters, min_parameters, max_parameters,
        parameter_names=None, steps=50):
    """
    Returns a `plotly.graph_objects.Figure` containing the data and simulation
    with interactive sliders.

    This function assumes the following keys naming convention:
        ids: '#ID'
        time: 'TIME in day'
        tumour volume: 'TUMOUR VOLUME in cm^3

    The axis labels as well as the hoverinfo assume that time is measured in
    day, volume is measured in cm^3.

    Arguments:
        data -- A pandas.DataFrame containing the measured time-series data of
                the tumour volume and the mass.
        model -- A `pints.ForwardModel`.
        parameters -- An array-like object with the model parameters for each
                      individual in the dataset.
                      Shape: (n_individuals, n_parameters)
    """
    # Check data has the correct type
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            'Data has to be pandas.DataFrame.')
    # Check that data has the required keys
    keys = ['#ID', 'TIME in day', 'TUMOUR VOLUME in cm^3', 'BODY WEIGHT in g']
    for key in keys:
        if key not in data.keys():
            raise ValueError(
                'Data must have key <' + str(key) +
                '>.')
    # Check that model has the correct type
    if not isinstance(model, pints.ForwardModel):
        raise TypeError(
            'Model needs to be an instance of `pints.ForwardModel`.')
    # Check that model has only one output dimension
    if model.n_outputs() != 1:
        raise ValueError(
            'Model output dimension has to be 1.')
    # Check that parameters have the correct dimensions
    default_parameters = np.asarray(default_parameters)
    if default_parameters.ndim != 1:
        raise ValueError(
            'Default parameters need to have dimension 1. `default_parameters`'
            ' has dimension <' + str(default_parameters.ndim) + '>.')
    min_parameters = np.asarray(min_parameters)
    if min_parameters.ndim != 1:
        raise ValueError(
            'Minimum parameters need to have dimension 1. `min_parameters` has'
            ' dimension <' + str(min_parameters.ndim) + '>.')
    max_parameters = np.asarray(max_parameters)
    if max_parameters.ndim != 1:
        raise ValueError(
            'Maximum parameters need to have dimension 1. `max_parameters` has'
            ' dimension <' + str(max_parameters.ndim) + '>.')
    # Check that parameters have the correct shape
    if default_parameters.shape != (model.n_parameters(),):
        raise ValueError(
            'Default parameters does not have the correct shape. Expected '
            'shape (n_parameters,) = ' + str((model.n_parameters(),)) + '.')
    if min_parameters.shape != (model.n_parameters(),):
        raise ValueError(
            'Minimum parameters do not have the correct shape. Expected shape '
            '(n_parameters,) = ' + str((model.n_parameters(),)) + '.')
    if max_parameters.shape != (model.n_parameters(),):
        raise ValueError(
            'Maximum parameters do not have the correct shape. Expected shape '
            '(n_parameters,) = ' + str((model.n_parameters(),)) + '.')

    if parameter_names is None:
        parameter_names = [
            'Parameter %d' % n for n in range(model.n_parameters())]

    # Check parameter names
    if len(parameter_names) != model.n_parameters():
        raise ValueError(
            'Number of parameter names does not match number of parameters.')

    # Define colorscheme
    n_ids = len(data['#ID'].unique())
    colors = plotly.colors.qualitative.Plotly[:n_ids]

    # Create figure
    fig = go.Figure()

    # Plot simulation for each slider step
    _add_slider_step_plots(
        fig, data, model, default_parameters, min_parameters, max_parameters,
        steps)

    # Scatter plot of control growth data
    _add_data(fig, data, colors)

    # Set X, Y axis and figure size
    fig.update_layout(
        autosize=True,
        xaxis_title=r'$\text{Time in day}$',
        yaxis_title=r'$\text{Tumour volume in cm}^3$',
        template="plotly_white")

    # Create parameter sliders
    sliders = _create_sliders(
        parameter_names, min_parameters, max_parameters, steps, data)

    # Add switch between linear and log y-scale
    fig.update_layout(
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"yaxis.type": "linear"}],
                        label="Linear y-scale",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.type": "log"}],
                        label="Log y-scale",
                        method="relayout"
                    )
                ]),
                pad={"r": 0, "t": -10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
            dict(
                type="buttons",
                direction="down",
                buttons=list([
                    dict(
                        args=[{
                            "sliders[0].visible": True,
                            "sliders[1].visible": False,
                            "sliders[2].visible": False}],
                        label="Initial volume",
                        method="relayout"
                    ),
                    dict(
                        args=[{
                            "sliders[0].visible": False,
                            "sliders[1].visible": True,
                            "sliders[2].visible": False}],
                        label="Exp. growth rate",
                        method="relayout"
                    ),
                    dict(
                        args=[{
                            "sliders[0].visible": False,
                            "sliders[1].visible": False,
                            "sliders[2].visible": True}],
                        label="Lin. growth rate",
                        method="relayout"
                    )
                ]),
                pad={"r": 0, "t": -10},
                showactive=True,
                x=1.07,
                xanchor="left",
                y=-0.1,
                yanchor="top"
            ),
        ]
    )

    return fig


def _add_data(fig, data, colors):
    """
    Adds a scatter plot of the data to the figure.
    """
    ids = data['#ID'].unique()
    for index, id_m in enumerate(ids):
        # Create mask for mouse
        mask = data['#ID'] == id_m

        # Get observed data for indiviudal
        observed_times = data['TIME in day'][mask].to_numpy()
        observed_data = data['TUMOUR VOLUME in cm^3'][mask]

        # Plot data
        fig.add_trace(go.Scatter(
            x=observed_times,
            y=observed_data,
            legendgroup="ID: %d" % id_m,
            name="ID: %d" % id_m,
            showlegend=True,
            hovertemplate=(
                "<b>Measurement </b><br>" +
                "ID: %d<br>" % id_m +
                "Time: %{x:} day<br>" +
                "Tumour volume: %{y:.02f} cm^3<br>" +
                "Cancer type: LXF A677<br>" +
                "<extra></extra>"),
            mode="markers",
            marker=dict(
                symbol='circle',
                opacity=0.7,
                line=dict(color='black', width=1),
                color=colors[index])
        ))


def _add_slider_step_plots(
        fig, data, model, parameters, min_parameters, max_parameters, steps):
    """
    Add a plot for each slider step to the figure.
    """
    # Define time range based on data
    start_experiment = data['TIME in day'].min()
    end_experiment = data['TIME in day'].max()
    simulated_times = np.linspace(
        start=start_experiment, stop=end_experiment)

    # Add plot for each slider step
    for param_id in range(model.n_parameters()):
        # Get default parameters
        params = parameters.copy()

        # Create line plot for each slider value
        for value_id, value in enumerate(np.linspace(
                start=min_parameters[param_id],
                stop=max_parameters[param_id], num=steps)):
            # Update parameters
            params[param_id] = value

            # Solve model
            simulated_data = model.simulate(params, simulated_times)

            # Show only first plot
            visible = False
            if param_id == 0 and value_id == 10:
                visible = True

            # Plot slider step
            fig.add_trace(go.Scatter(
                x=simulated_times,
                y=simulated_data,
                legendgroup="Model",
                name="Model",
                showlegend=True,
                hovertemplate=(
                    "<b>Simulation </b><br>" +
                    "Time: %{x:.0f} day<br>" +
                    "Tumour volume: %{y:.02f} cm^3<br>" +
                    "Cancer type: LXF A677<br>" +
                    "<br>" +
                    "<b>Parameters </b><br>" +
                    "Initial tumour volume: %.02f cm^3<br>" % params[0] +
                    "Exp. growth rate: %.02f 1/day<br>" % params[1] +
                    "Lin. growth rate: %.02f cm^3/day<br>" % params[2] +
                    "<extra></extra>"),
                visible=visible,
                mode="lines",
                line=dict(color='Black')
            ))


def _create_sliders(
        parameter_names, min_parameters, max_parameters, steps, data):
    """
    Returns slider objects that can be used in plotly's `update_layout`.
    """
    # Get number of parameters and measured individuals
    n_params = len(parameter_names)
    n_ids = len(data['#ID'].unique())

    # Create slider object for each parameter
    sliders = []
    for param_id in range(n_params):
        # Compute parameter values
        values = np.linspace(
            start=min_parameters[param_id], stop=max_parameters[param_id],
            num=steps)

        # Display plot for slider position and data
        slider_steps = []
        for step in range(steps):
            # All simulations False and data True
            s = dict(
                method="update",
                args=[{
                    "visible": [False] * (n_params * steps) +
                    [True] * n_ids}],
                label='%.02f' % values[step])

            # Make correct simulation visible
            s["args"][0]["visible"][param_id * steps + step] = True

            # Safe slider steps
            slider_steps.append(s)

        # Save slider
        sliders.append(
            dict(
                visible=True if param_id == 0 else False,
                active=10,
                currentvalue=dict(
                    prefix=parameter_names[param_id] + ': '),
                pad={"t": 50},
                steps=slider_steps))

    return sliders
