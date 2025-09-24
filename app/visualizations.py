"""Reusable Plotly visualizations for the interactive AI math explorer app."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import plotly.graph_objects as go


# --- Machine learning: linear regression plane ---------------------------------


@lru_cache(maxsize=1)
def _regression_dataset(
    n_points: int = 80,
    noise: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate and cache a synthetic linear regression dataset.

    The random seed is fixed so that the dataset remains stable across callback
    executions, keeping the interaction smooth and reproducible.
    """

    rng = np.random.default_rng(seed=21)
    x1 = rng.uniform(-4.0, 4.0, size=n_points)
    x2 = rng.uniform(-4.0, 4.0, size=n_points)
    true_weights = np.array([1.8, -2.4])
    true_bias = -0.2
    y = (
        true_weights[0] * x1
        + true_weights[1] * x2
        + true_bias
        + rng.normal(0.0, noise, size=n_points)
    )
    return x1, x2, y


def create_linear_regression_figure(
    weight_x1: float,
    weight_x2: float,
    bias: float,
) -> go.Figure:
    """Return an interactive 3D view of a regression plane versus the data."""

    x1, x2, y = _regression_dataset()

    x_range = np.linspace(x1.min() - 1.0, x1.max() + 1.0, 40)
    y_range = np.linspace(x2.min() - 1.0, x2.max() + 1.0, 40)
    grid_x1, grid_x2 = np.meshgrid(x_range, y_range)
    plane = weight_x1 * grid_x1 + weight_x2 * grid_x2 + bias

    predictions = weight_x1 * x1 + weight_x2 * x2 + bias
    residuals = y - predictions
    residual_magnitude = np.abs(residuals)

    scatter = go.Scatter3d(
        x=x1,
        y=x2,
        z=y,
        mode="markers",
        marker=dict(
            size=6,
            color=residual_magnitude,
            colorscale="Turbo",
            colorbar=dict(title="|residual|"),
            opacity=0.8,
        ),
        hovertemplate=
        "x₁: %{x:.2f}<br>x₂: %{y:.2f}<br>target: %{z:.2f}<br>"
        "residual: %{marker.color:.2f}<extra></extra>",
        name="observations",
    )

    plane_surface = go.Surface(
        x=grid_x1,
        y=grid_x2,
        z=plane,
        opacity=0.68,
        colorscale="Viridis",
        showscale=False,
        name="regression plane",
    )

    fig = go.Figure(data=[plane_surface, scatter])
    fig.update_layout(
        title="Interactive linear regression",
        scene=dict(
            xaxis_title="feature x₁",
            yaxis_title="feature x₂",
            zaxis_title="prediction",
            aspectmode="cube",
            xaxis=dict(backgroundcolor="rgba(10,10,30,0.15)"),
            yaxis=dict(backgroundcolor="rgba(10,10,30,0.15)"),
            zaxis=dict(backgroundcolor="rgba(10,10,30,0.05)"),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_dark",
    )
    return fig


# --- Neural network: feed-forward surface approximation ------------------------


@lru_cache(maxsize=1)
def _network_weights(max_hidden: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic base weights for a tiny fully-connected network."""

    rng = np.random.default_rng(seed=7)
    w1 = rng.normal(scale=0.9, size=(2, max_hidden))
    b1 = rng.normal(scale=0.4, size=(max_hidden,))
    w2 = rng.normal(scale=0.6, size=(max_hidden,))
    return w1, b1, w2


def _target_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth target function that the neural net aims to mimic."""

    return np.sin(x) * np.cos(y) + 0.25 * np.cos(2 * x + y)


def create_neural_network_figure(
    hidden_units: int,
    activation_scale: float,
    output_scale: float,
) -> go.Figure:
    """Render the neural network output surface alongside the target function."""

    w1, b1, w2 = _network_weights()
    grid = np.linspace(-2.5, 2.5, 60)
    mesh_x, mesh_y = np.meshgrid(grid, grid)
    flat_inputs = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)

    w1_scaled = w1[:, :hidden_units] * activation_scale
    b1_scaled = b1[:hidden_units] * activation_scale
    hidden = np.tanh(flat_inputs @ w1_scaled + b1_scaled)
    predictions = hidden @ (w2[:hidden_units] * output_scale)
    surface = predictions.reshape(mesh_x.shape)

    target = _target_surface(mesh_x, mesh_y)

    network_surface = go.Surface(
        x=mesh_x,
        y=mesh_y,
        z=surface,
        colorscale="Inferno",
        opacity=0.92,
        showscale=False,
        name="network output",
    )

    target_surface = go.Surface(
        x=mesh_x,
        y=mesh_y,
        z=target,
        colorscale="Blues",
        showscale=False,
        opacity=0.35,
        name="target function",
    )

    fig = go.Figure(data=[target_surface, network_surface])
    fig.update_layout(
        title="Neural network function approximation",
        scene=dict(
            xaxis_title="input x₁",
            yaxis_title="input x₂",
            zaxis_title="output",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_dark",
    )
    return fig


# --- Transformers: attention distribution --------------------------------------


@lru_cache(maxsize=1)
def _attention_components() -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    """Create deterministic token embeddings and projection matrices."""

    tokens = (
        "[CLS]",
        "A",
        "curious",
        "robot",
        "learns",
        "math",
        "today",
        "[SEP]",
    )
    rng = np.random.default_rng(seed=19)
    base_embeddings = rng.normal(scale=0.9, size=(len(tokens), 5))
    wq = rng.normal(scale=0.7, size=(5, 5))
    wk = rng.normal(scale=0.7, size=(5, 5))

    queries = base_embeddings @ wq
    keys = base_embeddings @ wk
    return tokens, queries, keys


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def get_attention_tokens() -> Tuple[str, ...]:
    """Return the ordered list of tokens used in the attention demo."""

    tokens, _, _ = _attention_components()
    return tokens


def create_attention_figure(temperature: float, focus_query: int) -> go.Figure:
    """Visualize scaled dot-product attention as a 3D surface."""

    tokens, queries, keys = _attention_components()
    scores = (queries @ keys.T) / temperature
    weights = _softmax(scores, axis=-1)

    x_indices = np.arange(len(tokens))
    y_indices = np.arange(len(tokens))
    mesh_x, mesh_y = np.meshgrid(x_indices, y_indices)
    hover_text = np.array(
        [
            [
                f"query: {tokens[q]}<br>key: {tokens[k]}"
                for k in x_indices
            ]
            for q in y_indices
        ]
    )

    surface = go.Surface(
        x=mesh_x,
        y=mesh_y,
        z=weights,
        colorscale="Viridis",
        opacity=0.9,
        name="attention weights",
        showscale=True,
        colorbar=dict(title="attention"),
        text=hover_text,
        hovertemplate="%{text}<br>weight: %{z:.3f}<extra></extra>",
    )

    highlight_row = go.Scatter3d(
        x=x_indices,
        y=np.full_like(x_indices, focus_query),
        z=weights[focus_query],
        mode="lines+markers",
        line=dict(color="#00F5FF", width=6),
        marker=dict(size=6, color="#00F5FF"),
        name=f"focus: {tokens[focus_query]}",
        text=[tokens[i] for i in x_indices],
        hovertemplate="key token: %{text}<br>attention: %{z:.3f}<extra></extra>",
    )

    fig = go.Figure(data=[surface, highlight_row])
    fig.update_layout(
        title="Transformer scaled dot-product attention",
        scene=dict(
            xaxis=dict(
                title="key positions",
                tickvals=x_indices,
                ticktext=tokens,
            ),
            yaxis=dict(
                title="query positions",
                tickvals=y_indices,
                ticktext=tokens,
            ),
            zaxis=dict(title="weight"),
            aspectratio=dict(x=1.1, y=1.1, z=0.6),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        template="plotly_dark",
    )
    return fig
