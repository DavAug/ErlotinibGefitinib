#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import plotly.colors
import plotly.graph_objects as go


def plot_optimised_parameters(parameters, scores, xlabels=None, ids=None):
    """
    Returns a `plotly.graph_objects.Figure` containing a box plot of the
    optimised parameter values and objective function scores for each
    individual and optimisation run.

    Arguments:
        parameters -- Array-like object with parameters for each individual and
                      each optimisation run.
                      Shape: (n_individuals, n_runs, n_parameters)
        scores -- Array-like object with objective function scores for each
                  individual and each optimisation run.
                  Shape: (n_individuals, n_runs)
        ids -- IDs of individuals. If `None` IDs are set to index.
    """
    parameters = np.asarray(parameters)
    scores = np.asarray(scores)
    if parameters.shape[0] != scores.shape[0]:
        raise ValueError(
            'Parameters and score do not have the same number of individuals.'
            'Shape parmeters <' + str(parameters.shape) + '>; shape scores <'
            + str(scores.shape) + '>.')
    if parameters.shape[1] != scores.shape[1]:
        raise ValueError(
            'Parameters and score do not have the same number of runs.'
            'Shape parmeters <' + str(parameters.shape) + '>; shape scores <'
            + str(scores.shape) + '>.')

    if xlabels is None:
        # Enumerate parameters and call score 'score'
        xlabels = [str(param_id) for param_id in range(parameters.shape[2])]
        xlabels += ['score']

    n_params = parameters.shape[2] + 1
    if len(xlabels) != parameters.shape[2]+1:
        raise ValueError(
            'Number of x labels does not match number of parameters plus '
            'score.')

    if ids is None:
        # Enumerate individuals
        ids = np.arange(parameters.shape[0])

    if len(ids) != parameters.shape[0]:
        raise ValueError(
            'Number of provided ids do not match number of individuals in '
            '`parameters` and `scores`.')

    # Define colorscheme
    colors = plotly.colors.qualitative.Plotly[:n_params]

    # Create figure
    fig = go.Figure()

    # Box plot of optimised model parameters
    n_ids = len(ids)
    for index in range(n_ids):
        # Get optimised parameters for individual
        params = parameters[index, ...]

        # Get scores
        score = scores[index, :]

        # Create box plot of for parameters
        for param_id in range(n_params-1):
            fig.add_trace(
                go.Box(
                    y=params[:, param_id],
                    name=xlabels[param_id],
                    boxpoints='all',
                    jitter=0.2,
                    pointpos=-1.5,
                    visible=True if index == 0 else False,
                    marker=dict(
                        symbol='circle',
                        opacity=0.7,
                        line=dict(color='black', width=1)),
                    marker_color=colors[param_id],
                    line_color=colors[param_id]))

        # Create box plot of for score
        fig.add_trace(
            go.Box(
                y=score,
                name=xlabels[-1],
                boxpoints='all',
                jitter=0.2,
                pointpos=-1.5,
                visible=True if index == 0 else False,
                marker=dict(
                    symbol='circle',
                    opacity=0.7,
                    line=dict(color='black', width=1)),
                marker_color=colors[-1],
                line_color=colors[-1]))

    # Set figure size
    fig.update_layout(
        autosize=True,
        template="plotly_white",
        yaxis_title="Estimates")

    # Add switch between individuals
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([dict(
                    args=[{"visible": [False] * (n_params * idx) +
                                      [True] * n_params +
                                      [False] * (n_params * (n_ids - idx - 1))}],
                    label="ID: %s" % str(ids[idx]),
                    method="restyle") for idx in range(n_ids)]),
                pad={"r": 0, "t": -10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )

    # Position legend
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=1.05))

    return fig
