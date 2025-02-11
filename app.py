from flask import Flask, render_template, request
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import numpy as np
import plotly.graph_objects as go

app = Flask(__name__)

# Load data and prepare coordinates
df = pd.read_csv('30.csv')  # Update path
layers = 7
all_embs = []
for i in range(1, layers+1):
    all_embs.extend([eval(emb) for emb in df[f"emb{i}"]])

n_neighbors = 15
affinity_matrix = kneighbors_graph(all_embs, n_neighbors=n_neighbors, mode='connectivity', include_self=False).toarray()
affinity_matrix = np.clip(affinity_matrix + affinity_matrix.T, a_min=0, a_max=1)
degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
laplacian_matrix = degree_matrix - affinity_matrix
eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
eig = sorted([(eigenvalues[i], eigenvectors[i]) for i in range(len(eigenvalues))], key=lambda x: x[0])
n_points = 256

x, y, z = {}, {}, {}
for i in range(1, 8):
    x[i] = []
    y[i] = []
    z[i] = []
    for j in range((i - 1) * n_points, i * n_points):
        x[i].append(eig[0][1][j])
        y[i].append(eig[1][1][j])
        z[i].append(eig[2][1][j])

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
    initial_colors, initial_text = get_colors_and_text(df['token'])
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
    for i in range(1, layers):
        for alpha in np.arange(0, 1, 0.05):
            xp, yp, zp = linear_combine(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1], alpha)
            colors, texts = get_colors_and_text(df['token'])
            
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

@app.route('/', methods=['GET', 'POST'])
def index():
    tokens = df['token']
    selected_flags = [0] * len(tokens)
    
    if request.method == 'POST':
        selected_indices = list(map(int, request.form.getlist('indices[]')))
        for idx in selected_indices:
            if 0 <= idx < len(tokens):
                selected_flags[idx] = 1
    else:
        selected_flags[0] = 1  # Default to first token

    return render_template('index.html',
                         tokens=enumerate(tokens),
                         selected_flags=selected_flags,
                         fig_html=generate_figure(selected_flags))

if __name__ == '__main__':
    app.run(debug=True)