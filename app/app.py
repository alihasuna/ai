"""Dash application presenting interactive 3D math visualizations for AI concepts."""
from __future__ import annotations

import dash
from dash import Dash, Input, Output, dcc, html

from . import visualizations

ATTENTION_TOKENS = visualizations.get_attention_tokens()

external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap",
]

app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "AI Math Explorer"

common_card_style = {
    "background": "rgba(15, 18, 51, 0.65)",
    "padding": "1.2rem",
    "borderRadius": "18px",
    "backdropFilter": "blur(6px)",
    "boxShadow": "0 18px 42px rgba(0, 0, 0, 0.35)",
    "border": "1px solid rgba(255, 255, 255, 0.08)",
}


def _slider_label(text: str) -> html.Div:
    return html.Div(text, style={"marginBottom": "0.35rem", "fontWeight": 500})


def _description(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "marginBottom": "0.9rem",
            "lineHeight": "1.6",
            "color": "#dbe7ff",
        },
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "AI Math Explorer",
                    style={
                        "fontSize": "2.6rem",
                        "letterSpacing": "0.04em",
                        "marginBottom": "0.4rem",
                    },
                ),
                html.P(
                    (
                        "Interactively explore the core mathematical ideas powering "
                        "modern AI systems. Adjust parameters to see how equations "
                        "shape model behaviour in real time."
                    ),
                    style={
                        "maxWidth": "780px",
                        "fontSize": "1.05rem",
                        "color": "#c7d7ff",
                        "lineHeight": "1.7",
                    },
                ),
            ],
            style={"marginBottom": "1.8rem"},
        ),
        dcc.Tabs(
            id="concept-tabs",
            value="ml",
            children=[
                dcc.Tab(
                    label="Machine Learning",
                    value="ml",
                    style={"background": "rgba(12,14,40,0.7)", "color": "#97b6ff"},
                    selected_style={
                        "background": "rgba(30,34,68,0.95)",
                        "color": "#ffffff",
                        "fontWeight": 600,
                    },
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        _description(
                                            (
                                                "Adjust the slope and intercept of a regression plane. Colors on the "
                                                "data points show the magnitude of residual errors, highlighting "
                                                "how optimisation aligns the plane with observations."
                                            )
                                        ),
                                        _slider_label("Weight for feature x₁"),
                                        dcc.Slider(
                                            id="ml-weight1",
                                            min=-3.0,
                                            max=3.0,
                                            step=0.1,
                                            value=1.8,
                                            marks={-3: "-3", 0: "0", 3: "3"},
                                        ),
                                        _slider_label("Weight for feature x₂"),
                                        dcc.Slider(
                                            id="ml-weight2",
                                            min=-3.0,
                                            max=3.0,
                                            step=0.1,
                                            value=-2.4,
                                            marks={-3: "-3", 0: "0", 3: "3"},
                                        ),
                                        _slider_label("Bias term"),
                                        dcc.Slider(
                                            id="ml-bias",
                                            min=-2.0,
                                            max=2.0,
                                            step=0.05,
                                            value=-0.2,
                                            marks={-2: "-2", 0: "0", 2: "2"},
                                        ),
                                    ],
                                    style={
                                        **common_card_style,
                                        "flex": "1 1 320px",
                                        "marginRight": "1.2rem",
                                    },
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="ml-graph",
                                        config={"displaylogo": False},
                                        style={"height": "520px"},
                                    ),
                                    style={"flex": "2 1 520px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "1.2rem",
                                "alignItems": "stretch",
                            },
                        )
                    ],
                ),
                dcc.Tab(
                    label="Neural Networks",
                    value="nn",
                    style={"background": "rgba(12,14,40,0.7)", "color": "#97b6ff"},
                    selected_style={
                        "background": "rgba(30,34,68,0.95)",
                        "color": "#ffffff",
                        "fontWeight": 600,
                    },
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        _description(
                                            (
                                                "Visualise how a shallow neural network stitches together nonlinear "
                                                "basis functions. Tune the number of hidden units, the activation "
                                                "sharpness and the output scaling to see the approximation evolve."
                                            )
                                        ),
                                        _slider_label("Hidden units"),
                                        dcc.Slider(
                                            id="nn-hidden",
                                            min=3,
                                            max=9,
                                            step=1,
                                            value=6,
                                            marks={i: str(i) for i in range(3, 10)},
                                        ),
                                        _slider_label("Activation scale"),
                                        dcc.Slider(
                                            id="nn-activation",
                                            min=0.6,
                                            max=2.6,
                                            step=0.1,
                                            value=1.3,
                                            marks={0.6: "0.6", 1.6: "1.6", 2.6: "2.6"},
                                        ),
                                        _slider_label("Output scale"),
                                        dcc.Slider(
                                            id="nn-output",
                                            min=0.3,
                                            max=2.0,
                                            step=0.05,
                                            value=1.0,
                                            marks={0.3: "0.3", 1.0: "1.0", 2.0: "2.0"},
                                        ),
                                    ],
                                    style={
                                        **common_card_style,
                                        "flex": "1 1 320px",
                                        "marginRight": "1.2rem",
                                    },
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="nn-graph",
                                        config={"displaylogo": False},
                                        style={"height": "520px"},
                                    ),
                                    style={"flex": "2 1 520px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "1.2rem",
                                "alignItems": "stretch",
                            },
                        )
                    ],
                ),
                dcc.Tab(
                    label="Transformers",
                    value="transformers",
                    style={"background": "rgba(12,14,40,0.7)", "color": "#97b6ff"},
                    selected_style={
                        "background": "rgba(30,34,68,0.95)",
                        "color": "#ffffff",
                        "fontWeight": 600,
                    },
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        _description(
                                            (
                                                "Scaled dot-product attention turns contextual similarity into a "
                                                "probability distribution. Alter the softmax temperature to see the "
                                                "focus sharpen or diffuse, and select a query token to track its row."
                                            )
                                        ),
                                        _slider_label("Softmax temperature"),
                                        dcc.Slider(
                                            id="attn-temperature",
                                            min=0.4,
                                            max=2.4,
                                            step=0.05,
                                            value=1.0,
                                            marks={0.4: "0.4", 1.0: "1.0", 2.4: "2.4"},
                                        ),
                                        _slider_label("Focused query token"),
                                        dcc.Slider(
                                            id="attn-query",
                                            min=0,
                                            max=len(ATTENTION_TOKENS) - 1,
                                            step=1,
                                            value=3,
                                            marks={
                                                i: token
                                                for i, token in enumerate(ATTENTION_TOKENS)
                                            },
                                        ),
                                    ],
                                    style={
                                        **common_card_style,
                                        "flex": "1 1 320px",
                                        "marginRight": "1.2rem",
                                    },
                                ),
                                html.Div(
                                    dcc.Graph(
                                        id="attn-graph",
                                        config={"displaylogo": False},
                                        style={"height": "520px"},
                                    ),
                                    style={"flex": "2 1 520px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "gap": "1.2rem",
                                "alignItems": "stretch",
                            },
                        )
                    ],
                ),
            ],
            style={
                "background": "rgba(22, 26, 66, 0.95)",
                "borderRadius": "24px",
                "overflow": "hidden",
                "boxShadow": "0 20px 45px rgba(0, 0, 0, 0.45)",
            },
        ),
    ],
    style={
        "background": "radial-gradient(circle at top, #1a1d4b, #080a1f 65%)",
        "color": "white",
        "minHeight": "100vh",
        "padding": "3rem",
        "fontFamily": "'Space Grotesk', sans-serif",
    },
)


@app.callback(
    Output("ml-graph", "figure"),
    Input("ml-weight1", "value"),
    Input("ml-weight2", "value"),
    Input("ml-bias", "value"),
)
def update_linear_regression(weight1: float, weight2: float, bias: float):
    return visualizations.create_linear_regression_figure(weight1, weight2, bias)


@app.callback(
    Output("nn-graph", "figure"),
    Input("nn-hidden", "value"),
    Input("nn-activation", "value"),
    Input("nn-output", "value"),
)
def update_neural_network(hidden_units: int, activation_scale: float, output_scale: float):
    return visualizations.create_neural_network_figure(
        hidden_units=int(hidden_units),
        activation_scale=activation_scale,
        output_scale=output_scale,
    )


@app.callback(
    Output("attn-graph", "figure"),
    Input("attn-temperature", "value"),
    Input("attn-query", "value"),
)
def update_attention(temperature: float, query_index: int):
    return visualizations.create_attention_figure(float(temperature), int(query_index))


if __name__ == "__main__":
    app.run(debug=True)
