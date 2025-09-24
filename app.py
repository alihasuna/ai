"""AI Math Explorer Streamlit app."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI Math Explorer",
    layout="wide",
    page_icon="🧠",
)


@st.cache_data
def generate_regression_data(seed: int, n_samples: int = 120):
    """Generate a reproducible synthetic regression dataset."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.5, 2.5, n_samples)
    x2 = rng.uniform(-2.5, 2.5, n_samples)
    noise = rng.normal(0, 0.6, n_samples)
    y = 1.5 * x1 - 0.8 * x2 + 2 + noise
    return x1, x2, y


def regression_figure(x1, x2, y, w1: float, w2: float, bias: float) -> go.Figure:
    """Create a 3D scatter plot with a regression plane."""
    grid = np.linspace(-2.5, 2.5, 30)
    gx, gy = np.meshgrid(grid, grid)
    plane = w1 * gx + w2 * gy + bias

    scatter = go.Scatter3d(
        x=x1,
        y=x2,
        z=y,
        mode="markers",
        marker=dict(
            size=5,
            color=y,
            colorscale="Viridis",
            opacity=0.8,
        ),
        name="Samples",
    )

    surface = go.Surface(
        x=gx,
        y=gy,
        z=plane,
        colorscale="Electric",
        opacity=0.7,
        showscale=False,
        name="Regression plane",
    )

    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            zaxis_title="Target",
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def neural_surface(weight_scale: float, bias_shift: float, activation: str) -> go.Figure:
    """Visualise how a tiny neural network sculpts a surface."""
    grid = np.linspace(-2.5, 2.5, 60)
    gx, gy = np.meshgrid(grid, grid)
    inputs = np.stack([gx, gy], axis=-1)

    base_weights = np.array(
        [
            [1.0, -1.0],
            [-1.5, 0.8],
            [0.7, 0.9],
        ]
    )
    base_bias = np.array([0.5, -0.3, 0.1])
    w1 = weight_scale * base_weights
    b1 = base_bias + bias_shift

    pre_act = inputs @ w1.T + b1
    if activation == "ReLU":
        hidden = np.maximum(pre_act, 0)
    elif activation == "tanh":
        hidden = np.tanh(pre_act)
    else:
        hidden = 1 / (1 + np.exp(-pre_act))

    w2 = np.array([1.2, -0.9, 0.7])
    b2 = -0.1
    output = hidden @ w2 + b2

    surface = go.Surface(
        x=gx,
        y=gy,
        z=output,
        colorscale="Magma",
        showscale=False,
        opacity=0.95,
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="Input x",
            yaxis_title="Input y",
            zaxis_title="Network output",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.1)),
        ),
    )
    return fig


def attention_weights(num_tokens: int, sharpness: float, shift: float) -> go.Figure:
    """Visualise a single-head self-attention pattern as a 3D surface."""
    positions = np.arange(num_tokens)
    weights = []
    for q in positions:
        center = q + shift * np.sin(q / max(num_tokens - 1, 1) * np.pi)
        diff = positions - center
        scores = np.exp(-sharpness * diff**2)
        norm = scores.sum()
        weights.append(scores / norm)
    weights = np.vstack(weights)

    gx, gy = np.meshgrid(positions, positions, indexing="ij")

    surface = go.Surface(
        x=gx,
        y=gy,
        z=weights,
        colorscale="IceFire",
        showscale=True,
        opacity=0.95,
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="Query position",
            yaxis_title="Key position",
            zaxis_title="Attention weight",
            xaxis=dict(nticks=num_tokens),
            yaxis=dict(nticks=num_tokens),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        ),
    )
    return fig


def main():
    st.title("🧠 AI Math Explorer")
    st.caption("Discover how foundational math concepts power modern AI models.")

    with st.sidebar:
        st.header("Choose a concept")
        section = st.radio(
            "Jump to:",
            (
                "Machine Learning Regression",
                "Neural Network Sculpting",
                "Transformer Attention",
            ),
        )
        st.markdown("""
        Use the controls below to tune the visualisations and observe how the
        underlying mathematics changes in real time.
        """)

    if section == "Machine Learning Regression":
        st.subheader("Machine Learning: Fitting a Plane to Data")
        st.write(
            """
            Linear regression searches for the plane that minimises the squared
            error between predictions and observed targets. Adjust the weights and
            bias to see how the regression surface responds to the data cloud.
            """
        )

        col1, col2, col3 = st.columns(3)
        seed = col1.slider("Random seed", 1, 999, 42, help="Regenerate synthetic data")
        w1 = col2.slider("Weight for feature 1", -2.5, 2.5, 1.5, 0.1)
        w2 = col3.slider("Weight for feature 2", -2.5, 2.5, -0.8, 0.1)
        bias = st.slider("Bias", -3.0, 3.0, 2.0, 0.1)

        x1, x2, y = generate_regression_data(seed)
        fig = regression_figure(x1, x2, y, w1, w2, bias)
        st.plotly_chart(fig, use_container_width=True)

    elif section == "Neural Network Sculpting":
        st.subheader("Neural Networks: Layered Function Sculptors")
        st.write(
            """
            Neural networks compose linear transforms with nonlinear activations
            to create richly expressive functions. The surface below shows how a
            miniature two-layer network carves the input space.
            """
        )

        col1, col2, col3 = st.columns(3)
        weight_scale = col1.slider("Weight scale", 0.2, 3.0, 1.0, 0.1)
        bias_shift = col2.slider("Bias shift", -1.0, 1.0, 0.0, 0.05)
        activation = col3.selectbox("Activation", ["ReLU", "tanh", "sigmoid"])

        fig = neural_surface(weight_scale, bias_shift, activation)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Interpretation:** Scaling the weights sharpens the ridges and valleys,
            while shifting the biases moves the regions where neurons activate. Each
            activation function shapes the surface differently—ReLU creates
            piecewise planar facets, tanh yields smooth waves, and sigmoid blends
            them softly.
            """
        )

    else:
        st.subheader("Transformers: Attention Landscapes")
        st.write(
            """
            Transformer models rely on attention to decide how strongly each token
            should look at every other token. Vary the sharpness and shift to see
            how attention patterns evolve across positions.
            """
        )

        col1, col2 = st.columns(2)
        num_tokens = col1.slider("Sequence length", 4, 16, 10)
        sharpness = col2.slider("Focus sharpness", 0.2, 3.0, 1.2, 0.1)
        shift = st.slider("Context shift", -1.5, 1.5, 0.4, 0.1)

        fig = attention_weights(num_tokens, sharpness, shift)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Tip:** Sharper focus values mimic low-temperature softmax behaviour,
            pushing attention toward a few tokens. The contextual shift imitates
            learned patterns such as looking ahead or behind in a sentence.
            """
        )

    st.markdown("---")
    st.write(
        "Designed for interactive teaching sessions—switch concepts from the sidebar"
        " and tweak the parameters to narrate the story of modern AI."
    )


if __name__ == "__main__":
    main()
