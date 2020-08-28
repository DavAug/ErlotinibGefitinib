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
            'Input data <' + str(data) + '> has to be pandas.DataFrame.')

    keys = ['#ID', 'TIME in day', 'TUMOUR VOLUME in cm^3', 'BODY WEIGHT in g']
    for key in keys:
        if key not in data.keys():
            raise ValueError(
                'Input data <' + str(data) + '> must have key <' + str(key) +
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

    This function assumes the follwing keys naming convention:
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
            'Data <' + str(data) + '> has to be pandas.DataFrame.')
    # Check that data has the required keys
    keys = ['#ID', 'TIME in day', 'TUMOUR VOLUME in cm^3', 'BODY WEIGHT in g']
    for key in keys:
        if key not in data.keys():
            raise ValueError(
                'Data <' + str(data) + '> must have key <' + str(key) +
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
