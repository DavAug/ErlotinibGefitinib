#
# This file is part of the ErlotinibGefitinib repository
# (https://github.com/DavAug/ErlotinibGefitinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#



def plot_error_model(
        data, struc_model, error_model, parameters, pooled_error=False):
    r"""
    Returns a `plotly.graph_objects.Figure` containing a measurements versus
    structural model predictions scatter plot and a residuals versus
    structural model predictions scatter plot.

    This function assumes the following keys naming convention for the data:
        ids: '#ID'
        time: 'TIME in day'
        tumour volume: 'TUMOUR VOLUME in cm^3
        mass: 'BODY WEIGHT in g'.

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
    # Import data
    # Get path of current working directory
    path = os.getcwd()

    # Import LXF A677 control growth data
    data = pd.read_csv(path + '/data/lxf_control_growth.csv')

    # Match observations with predictions
    data = data.merge(structural_model_predictions, on=['#ID', 'TIME in day'])

    # Get noise parameters
    sigmas = median_parameters[:, -1]

    # Get mouse ids
    mouse_ids = data['#ID'].unique()

    # Get number of mice
    n_mice = len(mouse_ids)

    # Define colorscheme
    colors = plotly.colors.qualitative.Plotly[:n_mice]

    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05)

    # Scatter plot LXF A677 time-series data for each mouse
    for index, id_m in enumerate(mouse_ids):
        # Create mask for mouse
        mask = data['#ID'] == id_m

        # Get predicted tumour volumes for mouse
        predicted_volumes = data['PREDICTED TUMOUR VOLUME in cm^3'][mask]

        # Get observed tumour volumes for mouse
        observed_volumes = data['TUMOUR VOLUME in cm^3'][mask]

        # Get noise parameter
        sigma = sigmas[index]

        # Plot I: Measured vs predicted volumes
        # Plot measured tumour volume versus structural model tumour volume
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=observed_volumes,
                legendgroup="Measurement",
                name="Measurement",
                showlegend=True,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>ID: %d</b><br>" % (id_m) +
                    "Structural model: %{x:.02f} cm^3<br>" +
                    "Measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=colors[index],
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=1,
            col=1)

        # Plot median of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=predicted_volumes,
                legendgroup="Error model",
                name="Error model",
                showlegend=True,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: Mean</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(color=colors[index]),
                opacity=0.7),
            row=1,
            col=1)

        # Plot 1-sigma interval of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=predicted_volumes + sigma,
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 1-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.7),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=predicted_volumes - sigma,
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 1-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.7),
            row=1,
            col=1)

        # Plot 2-sigma interval of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=predicted_volumes + 2 * sigma,
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 2-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black',
                    width=1.5),
                opacity=0.5),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=predicted_volumes - 2 * sigma,
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 2-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black',
                    width=1.5),
                opacity=0.5),
            row=1,
            col=1)

        # Plot 3-sigma interval of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=predicted_volumes + 3 * sigma,
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 3-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black',
                    width=1),
                opacity=0.3),
            row=1,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=predicted_volumes - 3 * sigma,
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 3-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black',
                    width=1),
                opacity=0.3),
            row=1,
            col=1)

        # Plot II: Residuals vs predicted volumes
        # Plot residuals versus structural model tumour volume
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=observed_volumes - predicted_volumes,
                legendgroup="Measurement",
                name="Measurement",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>ID: %d</b><br>" % (id_m) +
                    "Structural model: %{x:.02f} cm^3<br>" +
                    "Residual: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=colors[index],
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=2,
            col=1)

        # Plot median of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=np.full(shape=len(predicted_volumes), fill_value=0),
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: Mean</b><br>" +
                    "Hypothetical residual: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(color=colors[index]),
                opacity=0.7),
            row=2,
            col=1)

        # Plot 1-sigma interval of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=np.full(shape=len(predicted_volumes), fill_value=sigma),
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 1-sigma interval</b><br>" +
                    "Hypothetical residual: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.7),
            row=2,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=np.full(shape=len(predicted_volumes), fill_value=-sigma),
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 1-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.7),
            row=2,
            col=1)

        # Plot 2-sigma interval of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=np.full(shape=len(predicted_volumes), fill_value=2 * sigma),
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 2-sigma interval</b><br>" +
                    "Hypothetical residual: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.5),
            row=2,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=np.full(shape=len(predicted_volumes), fill_value=-2 * sigma),
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 2-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.5),
            row=2,
            col=1)

        # Plot 3-sigma interval of error model
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=np.full(shape=len(predicted_volumes), fill_value=3 * sigma),
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 3-sigma interval</b><br>" +
                    "Hypothetical residual: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.3),
            row=2,
            col=1)
        fig.add_trace(
            go.Scatter(
                x=predicted_volumes,
                y=np.full(shape=len(predicted_volumes), fill_value=-3 * sigma),
                legendgroup="Error model",
                name="Error model",
                showlegend=False,
                visible=True if index == 0 else False,
                hovertemplate=
                    "<b>Error model: 3-sigma interval</b><br>" +
                    "Hypothetical measurement: %{y:.02f} cm^3<br>" +
                    "Cancer type: Lung cancer (LXF A677)<br>" +
                    "<extra></extra>",
                mode="lines",
                line=dict(
                    color='Black'),
                opacity=0.3),
            row=2,
            col=1)

    # Set figure size
    fig.update_layout(
        autosize=True,
        template="plotly_white")

    # Set X and Y axes
    fig.update_xaxes(title_text=r'$\text{Structural model predictions in cm}^3$', row=2, col=1)
    fig.update_yaxes(title_text=r'$\text{Tumour volume in cm}^3$', row=1, col=1)
    fig.update_yaxes(title_text=r'$\text{Residuals in cm}^3$', row=2, col=1)


    # Add switch between mice
    fig.update_layout(
        updatemenus=[
            dict(
                type = "buttons",
                direction = "right",
                buttons=list([
                    dict(
                        args=[{"visible": [True]*(8 * 2) + [False]*(8 * 2 * 7)}],
                        label="ID: %d" % mouse_ids[0],
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False]*(8 * 2) + [True]*(8 * 2) + [False]*(8 * 2 * 6)}],
                        label="ID: %d" % mouse_ids[1],
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False]*(8 * 2 * 2) + [True]*(8 * 2) + [False]*(8 * 2 * 5)}],
                        label="ID: %d" % mouse_ids[2],
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False]*(8 * 2 * 3) + [True]*(8 * 2) + [False]*(8 * 2 * 4)}],
                        label="ID: %d" % mouse_ids[3],
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False]*(8 * 2 * 4) + [True]*(8 * 2) + [False]*(8 * 2 * 3)}],
                        label="ID: %d" % mouse_ids[4],
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False]*(8 * 2 * 5) + [True]*(8 * 2) + [False]*(8 * 2 * 2)}],
                        label="ID: %d" % mouse_ids[5],
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False]*(8 * 2 * 6) + [True]*(8 * 2) + [False]*(8 * 2)}],
                        label="ID: %d" % mouse_ids[6],
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False]*(8 * 2 * 7) + [True]*(8 * 2)}],
                        label="ID: %d" % mouse_ids[7],
                        method="restyle"
                    )
                ]),
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


def _predict_tumour_volume():
    # Get mouse IDs and times
    mouse_ids_and_times = data[['#ID', 'TIME in day']]

    # Get median parameters for each mouse
    median_parameters = np.median(transf_params, axis=1)

    # Instantiate model
    model = DimensionlessLogTransformedPintsModel()

    # Create container for simulated synthesised data
    structural_model_predictions = []

    # Simulate "noise-free" data
    mouse_ids = mouse_ids_and_times['#ID'].unique()
    for index, mouse_id in enumerate(mouse_ids):
        # Create mask for mouse
        mask = mouse_ids_and_times['#ID'] == mouse_id

        # Get times
        times = mouse_ids_and_times[mask]['TIME in day'].to_numpy() / characteristic_time

        # Get parameters
        parameters = median_parameters[index, :n_structural_params]

        # Predict volumes
        predicted_volumes = model.simulate(parameters, times) * characteristic_volume

        # Save dataframe
        df = pd.DataFrame({
            '#ID': mouse_ids_and_times[mask]['#ID'],
            'TIME in day': mouse_ids_and_times[mask]['TIME in day'],
            'PREDICTED TUMOUR VOLUME in cm^3': predicted_volumes})
        structural_model_predictions.append(df)

    # Merge mouse dataframes to one dataframe
    structural_model_predictions = pd.concat(structural_model_predictions)