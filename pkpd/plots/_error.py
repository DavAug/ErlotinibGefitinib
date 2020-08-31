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
from plotly.subplots import make_subplots


ALLOWED_ERROR_MODELS = [
        'constant Gaussian', 'multiplicative Gaussian', 'combined Gaussian']
NUMBER_ERROR_PARAMETERS = {
    'constant Gaussian': 1,
    'multiplicative Gaussian': 1,
    'combined Gaussian': 2}


def _add_error_model(fig, predictions, sigma, idx, color, visible):
    """
    Adds the median, 1-sigma, 2-sigma, and 3-sigma intervals of a Gaussian
    error model to the figure.
    """
    # Plot median of error model
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=predictions,
            legendgroup="Error model",
            name="Error model",
            showlegend=True,
            visible=visible,
            hovertemplate=(
                "<b>Error model: Mean</b><br>"
                "Predicted measurement: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="lines",
            line=dict(color=color),
            opacity=0.7),
        row=1,
        col=1)

    # Plot 1-sigma interval of error model
    _add_sigma_interval_predictions(
        fig=fig, predictions=predictions, sigma=sigma, number=1, opacity=0.7,
        width=2, visible=visible)

    # Plot 2-sigma interval of error model
    _add_sigma_interval_predictions(
        fig=fig, predictions=predictions, sigma=sigma, number=2, opacity=0.5,
        width=1.5, visible=visible)

    # Plot 3-sigma interval of error model
    _add_sigma_interval_predictions(
        fig=fig, predictions=predictions, sigma=sigma, number=3, opacity=0.3,
        width=1, visible=visible)


def _add_measurements_versus_predictions_plot(
        fig, measurements, predictions, idx, color, visible):
    """
    Adds a measurements versus predictions plot for individual `idx` to the
    figure.
    """
    # Plot measured tumour volume versus structural model tumour volume
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=measurements,
            legendgroup="Measurement",
            name="Measurement",
            showlegend=True,
            visible=visible,
            hovertemplate=(
                "<b>ID: %d</b><br>" % idx +
                "Structural model: %{x:.02f} cm^3<br>"
                "Measurement: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="markers",
            marker=dict(
                symbol='circle',
                color=color,
                opacity=0.7,
                line=dict(color='black', width=1))),
        row=1,
        col=1)


def _add_residual_error_model(fig, predictions, sigma, idx, color, visible):
    """
    Adds the residual error model to the figure.
    """
    # Plot median of error model
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=np.full(shape=len(predictions), fill_value=0),
            legendgroup="Error model",
            name="Error model",
            showlegend=False,
            visible=visible,
            hovertemplate=(
                "<b>Error model: Mean</b><br>"
                "Hypothetical residual: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="lines",
            line=dict(color=color),
            opacity=0.7),
        row=2,
        col=1)

    # Plot 1-sigma interval of error model
    _add_sigma_interval_residuals(
        fig=fig, predictions=predictions, sigma=sigma, number=1, opacity=0.7,
        width=2, visible=visible)

    # Plot 2-sigma interval of error model
    _add_sigma_interval_residuals(
        fig=fig, predictions=predictions, sigma=sigma, number=2, opacity=0.5,
        width=1.5, visible=visible)

    # Plot 3-sigma interval of error model
    _add_sigma_interval_residuals(
        fig=fig, predictions=predictions, sigma=sigma, number=3, opacity=0.3,
        width=1, visible=visible)


def _add_residuals_versus_predictions_plot(
        fig, measurements, predictions, idx, color, visible):
    """
    Adds a residuals versus predictions plot for individual `idx` to the
    figure.
    """
    # Plot residuals versus structural model tumour volume
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=measurements - predictions,
            legendgroup="Measurement",
            name="Measurement",
            showlegend=False,
            visible=visible,
            hovertemplate=(
                "<b>ID: %d</b><br>" % idx +
                "Structural model: %{x:.02f} cm^3<br>"
                "Residual: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="markers",
            marker=dict(
                symbol='circle',
                color=color,
                opacity=0.7,
                line=dict(color='black', width=1))),
        row=2,
        col=1)


def _add_sigma_interval_predictions(
        fig, predictions, sigma, number, opacity, width, visible):
    """
    Adds `number`-sigma interval lines of the error model to the measurement
    versus predictions plot.
    """
    # Add upper limit of interval
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=predictions + number * sigma,
            legendgroup="Error model",
            name="Error model",
            showlegend=False,
            visible=visible,
            hovertemplate=(
                "<b>Error model: %d-sigma interval</b><br>" % number +
                "Predicted measurement: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="lines",
            line=dict(
                color='Black',
                width=width),
            opacity=opacity),
        row=1,
        col=1)

    # Add lower limit of interval
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=predictions - number * sigma,
            legendgroup="Error model",
            name="Error model",
            showlegend=False,
            visible=visible,
            hovertemplate=(
                "<b>Error model: %d-sigma interval</b><br>" % number +
                "Hypothetical measurement: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="lines",
            line=dict(
                color='Black',
                width=width),
            opacity=opacity),
        row=1,
        col=1)


def _add_sigma_interval_residuals(
        fig, predictions, sigma, number, opacity, width, visible):
    """
    Adds `number`-sigma interval lines of the error model to the residuals
    versus predictions plot.
    """
    # Add upper limit of interval
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=np.full(shape=len(predictions), fill_value=number * sigma),
            legendgroup="Error model",
            name="Error model",
            showlegend=False,
            visible=visible,
            hovertemplate=(
                "<b>Error model: %d-sigma interval</b><br>" % number +
                "Hypothetical residual: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="lines",
            line=dict(
                color='Black',
                width=width),
            opacity=opacity),
        row=2,
        col=1)

    # Add lower limit of interval
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=np.full(shape=len(predictions), fill_value=-number * sigma),
            legendgroup="Error model",
            name="Error model",
            showlegend=False,
            visible=visible,
            hovertemplate=(
                "<b>Error model: %d-sigma interval</b><br>" % number +
                "Hypothetical residual: %{y:.02f} cm^3<br>"
                "Cancer type: Lung cancer (LXF A677)<br>"
                "<extra></extra>"),
            mode="lines",
            line=dict(
                color='Black',
                width=width),
            opacity=opacity),
        row=2,
        col=1)


def _compute_standard_deviation(error_model, predictions, parameters):
    """
    Returns the standard deviation of the error model for each prediction.
    """
    if error_model == 'constant Gaussian':
        sigma = parameters[0]
        return np.full(shape=len(predictions), fill_value=sigma)
    if error_model == 'multiplicative Gaussian':
        sigma = parameters[0]
        return predictions * sigma
    if error_model == 'multiplicative Gaussian':
        sigma_base = parameters[0]
        sigma_rel = parameters[1]
        return sigma_base + predictions * sigma_rel


def plot_measurements_and_error_model(
        data, struc_model, error_model, parameters, pooled_error=False):
    r"""
    Returns a `plotly.graph_objects.Figure` containing a measurements versus
    structural model predictions scatter plot and a residuals versus
    structural model predictions scatter plot.

    This function assumes the following keys naming convention for the data:
        ids: '#ID'
        time: 'TIME in day'
        tumour volume: 'TUMOUR VOLUME in cm^3.

    The axis labels as well as the hoverinfo assume that time is measured in
    day and volume is measured in cm^3.

    The `error_model` is one of three keywords specifying the error model used
    to model the deviations of the data from the structural model

    .. math::
        \varepsilon = V^{\text{obs}} - V^s_T,

    where :math:`V^{\text{obs}}` are the measured tumour volumes and
    :math:`V^s_T` the structural model predictions.

    The implemented error models are

    .. 'constant Gaussian': Parameters = :math:`\sigma`
    .. math::
        \varepsilon = \sigma \epsilon _n, \quad \text{where} \quad
            \epsilon _n \sim \mathcal{N}(0, 1),

    .. 'multiplicative Gaussian': Parameters = :math:`\sigma _{\text{rel}}`
    .. math::
        \varepsilon = \sigma _{\text{rel}}V^s_T \epsilon _n, \quad \text{where}
            \quad \epsilon _n \sim \mathcal{N}(0, 1),

    .. 'combined Gaussian': Parameters =
        :math:`(\sigma _{\text{base}}, \sigma _{\text{rel}})`
    .. math::
        \varepsilon = \sigma _{\text{base}} + \sigma _{\text{rel}}V^s_T
            \epsilon _n, \quad \text{where} \quad
            \epsilon _n \sim \mathcal{N}(0, 1),

     Arguments:
        data -- A pandas.DataFrame containing the measured time-series data of
                the tumour volume.
        struc_model -- A `pints.ForwardModel` implementing the structural
                       tumour growth.
        error_model -- One of three error models: ['constant Gaussian',
                       'multiplicative Gaussian', 'combined Gaussian'].
        parameters -- An array-like object with the model parameters for each
                      individual in the dataset. The structural model
                      parameters are assumed to come before the error model
                      parameters.
                      Shape: (n_individuals, n_parameters)
        pooled_error -- A boolean flag indicating whether the error across
                        individuals is plotted in one figure (True) or in
                        separate figures (False).
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
    if not isinstance(struc_model, pints.ForwardModel):
        raise TypeError(
            'Structural model `struc_model` needs to be an instance of '
            '`pints.ForwardModel`.')
    # Check that model has only one output dimension
    if struc_model.n_outputs() != 1:
        raise ValueError(
            'Structural model output dimension has to be 1.')
    # Check that error model is valid
    if error_model not in ALLOWED_ERROR_MODELS:
        raise ValueError(
            'Error model <' + str(error_model) + '> is not an allowed error '
            'model. Allowed error models are <' + str(ALLOWED_ERROR_MODELS)
            + '>.')

    # Get number of individuals
    n_ids = len(data['#ID'].unique())

    # Check that parameters have the correct shape
    n_struc_params = struc_model.n_parameters()
    n_error_params = NUMBER_ERROR_PARAMETERS[error_model]
    parameters = np.asarray(parameters)
    if parameters.shape != (n_ids, n_struc_params + n_error_params):
        raise ValueError(
            'Parameters does not have the correct shape. Expected shape '
            '(n_individuals, n_parameters) = ' +
            str((n_ids, n_struc_params + n_error_params)) + '.')
    # Check that the error model parameters across individuals are identical,
    # if the error model is pooled.
    if pooled_error:
        params = parameters[:, -n_error_params:]
        if not np.all(params == params[0, :]):
            raise ValueError(
                'Pooling of the error model makes only sense if the error '
                'model parameters are the same across all individuals.')

    # Define colorscheme
    colors = plotly.colors.qualitative.Plotly[:n_ids]

    # Create figure
    n_subplots = 2
    fig = make_subplots(
        rows=n_subplots, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
        vertical_spacing=0.05)

    # Create plots
    ids = data['#ID'].unique()
    for index, idx in enumerate(ids):
        # Create mask for mouse
        mask = data['#ID'] == idx

        # Predict tumour volumes
        params = parameters[index, :n_struc_params]
        observed_times = data['TIME in day'][mask]
        predicted_volumes = struc_model.simulate(
            params, observed_times.to_numpy())

        # Get observed tumour volumes for mouse
        observed_volumes = data['TUMOUR VOLUME in cm^3'][mask]

        # Get noise parameter
        params = parameters[index, -n_error_params:]
        sigma = _compute_standard_deviation(
            error_model, predicted_volumes, params)

        # Plot measurements versus predictions / residuals versus predictions
        visible = True if index == 0 else False
        if pooled_error:
            # Pooling the error model results in plotting all individuals in
            # the same plot.
            visible = True
        _add_measurements_versus_predictions_plot(
            fig, observed_volumes, predicted_volumes, idx, colors[index],
            visible)
        _add_residuals_versus_predictions_plot(
            fig, observed_volumes, predicted_volumes, idx, colors[index],
            visible)

        # Plot error model only once if error is pooled, else for each
        # individual
        color = colors[index]
        if pooled_error:
            # Make colour of error neutral
            color = 'black'
        if index == 0:
            _add_error_model(
                fig, predicted_volumes, sigma, idx, color, True)
            _add_residual_error_model(
                fig, predicted_volumes, sigma, idx, color, visible)
        elif not pooled_error:
            # Hide error for other individuals at first
            _add_error_model(
                fig, predicted_volumes, sigma, idx, color, False)
            _add_residual_error_model(
                fig, predicted_volumes, sigma, idx, color, False)

    # Set figure size
    fig.update_layout(
        autosize=True,
        template="plotly_white")

    # Set X and Y axes
    fig.update_xaxes(
        title_text=r'$\text{Structural model predictions in cm}^3$',
        row=2, col=1)
    fig.update_yaxes(
        title_text=r'$\text{Tumour volume in cm}^3$', row=1, col=1)
    fig.update_yaxes(
        title_text=r'$\text{Residuals in cm}^3$', row=2, col=1)

    if not pooled_error:
        # Number of graph objects per subplot
        # [data, mean, 2 times 1-, 2-, 3-sigma interval]
        n_go = 8

        # Add switches between individuals
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=list([dict(
                        args=[{
                            "visible": [False] * (n_go * n_subplots * idx) +
                            [True] * (n_go * n_subplots) +
                            [False] * (n_go * n_subplots * (n_ids - idx - 1))}
                            ],
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
