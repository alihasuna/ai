# AI Math Explorer

An interactive Streamlit application that explains core mathematical ideas behind
machine learning, neural networks, and transformer attention through elegant 3D
visualisations.

## Features

- **Machine Learning Regression:** manipulate linear regression weights to see
  how a fitted plane aligns with data.
- **Neural Network Sculpting:** explore how layered nonlinearities shape a
  function landscape.
- **Transformer Attention:** tune the sharpness and contextual shift of an
  attention map to understand token-to-token interactions.

## Getting Started

1. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open the provided local URL in your browser and interact with the sidebar to
   switch between concepts and adjust parameters.

## Project Structure

```
.
├── app.py            # Streamlit application entry point
├── README.md         # Project documentation
└── requirements.txt  # Python dependencies
```

## License

This project is released under the MIT License.
