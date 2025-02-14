from flask import Flask, render_template, request
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

app = Flask(__name__)
num_layers = 7
emb_dim = 384

def load_model():
    ckpt_path = os.path.join('out-tinystories', 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))  # Ensure model loads on CPU
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    model.to("cpu")  # Move model to CPU
    state_dict = checkpoint['model']

    # Remove unwanted prefixes in state dict
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode
    model.to("cpu")  # Move model to CPU explicitly

    return model

def get_encode_decode():
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode

print("Loading model...")
model = load_model()
print("Model loaded. Loading encode and decode functions...")
encode, decode = get_encode_decode()
print("Encode and decode loaded. Loading precomputed data...")

def get_embeddings(text):
    start_ids = encode(text)
    x = torch.tensor(start_ids, dtype=torch.long, device="cpu")[None, ...]

    embs = model.get_embeddings(x)
    return embs, [decode([token]) for token in start_ids]

def load_precomputed_data():
    # Load data and prepare coordinates
    df = pd.read_csv('30.csv')  # Update path
    all_embs = []
    for i in range(1, num_layers + 1):
        all_embs.append([eval(emb) for emb in df[f"emb{i}"]])
    return np.array(all_embs), df["token"]

def reduce_3d(all_embs):
    n_neighbors = 15
    n_points = len(all_embs[0])
    all_embs_reshaped = all_embs.reshape(n_points * num_layers, emb_dim)
    affinity_matrix = kneighbors_graph(all_embs_reshaped, n_neighbors=n_neighbors, mode='connectivity', include_self=False).toarray()
    affinity_matrix = np.clip(affinity_matrix + affinity_matrix.T, a_min=0, a_max=1)
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    laplacian_matrix = degree_matrix - affinity_matrix
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    eig = sorted([(eigenvalues[i], eigenvectors[i]) for i in range(len(eigenvalues))], key=lambda x: x[0])

    x, y, z = {}, {}, {}
    for i in range(1, num_layers + 1):
        x[i] = []
        y[i] = []
        z[i] = []
        for j in range((i - 1) * n_points, i * n_points):
            x[i].append(eig[0][1][j])
            y[i].append(eig[1][1][j])
            z[i].append(eig[2][1][j])
    
    return x, y, z

all_embs, tokens = load_precomputed_data()
print("Loaded precomputed data. Computing 3d reductions...")
x, y, z = reduce_3d(all_embs)
print("Computed 3D reductions.")

def linear_combine(x1, y1, z1, x2, y2, z2, alpha):
    return (
        [x1[i] * (1 - alpha) + x2[i] * alpha for i in range(len(x1))],
        [y1[i] * (1 - alpha) + y2[i] * alpha for i in range(len(y1))],
        [z1[i] * (1 - alpha) + z2[i] * alpha for i in range(len(z1))]
    )

def generate_figure(selected_flags):

    def get_colors_and_text(df_tokens):
        colors = []
        texts = []
        for i, t in enumerate(df_tokens):
            if selected_flags[i]:
                colors.append('rgba(255, 165, 0, 1)')
                texts.append(t)
            else:
                colors.append('rgba(0, 0, 255, 0.1)')
                texts.append('')
        return colors, texts

    # Rest of the figure generation code remains the same
    initial_colors, initial_text = get_colors_and_text(tokens)
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x[1],
            y=y[1],
            z=z[1],
            mode='markers+text',
            marker=dict(size=5, color=initial_colors),
            text=initial_text,
            textposition='top center',
            textfont=dict(size=15)
        )],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(nticks=10, range=[-0.1, 0.1]),
                yaxis=dict(nticks=10, range=[-0.1, 0.1]),
                zaxis=dict(nticks=10, range=[-0.1, 0.1]),
                aspectmode='cube'
            ),
            margin=dict(r=20, b=10, l=10, t=50),
            scene_camera=dict(eye=dict(x=1, y=1, z=1)),
            width=1200,
            height=800,
        )
    )

    # Animation frames code remains the same
    frames = []
    for i in range(1, num_layers):
        for alpha in np.arange(0, 1, 0.05):
            xp, yp, zp = linear_combine(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1], alpha)
            colors, texts = get_colors_and_text(tokens)
            
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=xp,
                    y=yp,
                    z=zp,
                    mode='markers+text',
                    marker=dict(size=5, color=colors),
                    text=texts,
                    textposition='top center',
                    textfont=dict(size=12)
                )],
                name=f'Layer {(i+alpha):.2f}',
                layout=go.Layout(title=dict(text=f"Layer {(i+alpha):.2f}"))
            ))
    
    fig.frames = frames

    # Animation controls code remains the same
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 30, "redraw": True},
                                  "transition": {"duration": 0},
                                  "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate"}])
            ],
            active=1
        )]
    )
    
    html = fig.to_html(full_html=False)
    html += """
    <script>
    document.querySelector('.plotly-graph-div').on('plotly_afterplot', function(){
        Plotly.animate(
            this,
            [],
            {frame: {duration: 0}, transition: {duration: 0}}
        );
    });
    </script>
    """
    return html

@app.route('/visualize', methods=['GET', 'POST'])
def index():

    selected_flags = [0] * len(tokens)
    
    if request.method == 'POST':
        selected_indices = list(map(int, request.form.getlist('indices[]')))
        for idx in selected_indices:
            if 0 <= idx < len(tokens):
                selected_flags[idx] = 1
    else:
        selected_flags[0] = 1  # Default to first token

    return render_template('visualize.html',
                         tokens=enumerate(tokens),
                         selected_flags=selected_flags,
                         fig_html=generate_figure(selected_flags))

if __name__ == '__main__':
    app.run(debug=True)