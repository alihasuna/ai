# AI Math Explorer

An interactive [Dash](https://dash.plotly.com/) web app that illustrates core
mathematical ideas behind three pillars of modern AI using elegant 3D
visualisations:

- **Machine learning** – manipulate a regression plane to understand how loss is
  driven by residuals.
- **Neural networks** – watch a shallow network combine nonlinear activations
  to approximate a target surface.
- **Transformers** – explore how scaled dot-product attention distributes focus
  across a sequence.

## Getting started

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Dash development server:
   ```bash
   python -m app.app
   ```
4. Open your browser at <http://127.0.0.1:8050> to interact with the
   visualisations.

The app is entirely client-side rendered, so you can modify the controls without
reloading the page. Each plot can be rotated, zoomed and panned for different
perspectives.

## Project structure

```
app/
├── __init__.py
├── app.py           # Dash entry point and layout
└── visualizations.py # Plotly figure factories for each concept
README.md
requirements.txt
```

## Development tips

- The figures are designed to be deterministic so callbacks respond smoothly
  without regenerating random data.
- Feel free to extend the app with more tabs or controls to dive deeper into
  additional AI maths topics.
