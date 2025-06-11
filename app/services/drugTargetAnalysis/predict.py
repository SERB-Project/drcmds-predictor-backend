import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
import sys
import base64
import io
# Add all necessary paths to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local modules
from app.services.drugTargetAnalysis.preprocess import smile_to_graph, seq_cat
# Import from gen.py - note the lowercase 'gen'
from gen import GEN
# Path to the model file
MODEL_PATH = "app/models/drugTargetAnalysis/drug_target_anaylsis.pt"

# Load the model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Import the GEN class from the gen module
    torch.serialization.add_safe_globals([GEN])
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
    return model, device

# Process a single SMILES string and target sequence
def process_input(compound_smiles, target_sequence):
    # Process compound
    c_size, features, edge_index = smile_to_graph(compound_smiles)
    
    # Convert to tensor
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index).t()
    
    # Create data object
    data = Data(x=x, edge_index=edge_index)
    
    # Process target sequence if provided
    if target_sequence:
        target_x = seq_cat(target_sequence)
        data.target = torch.LongTensor(target_x)
    else:
        # Use zeros if no target sequence
        data.target = torch.zeros(1000, dtype=torch.long)
    
    # Add batch attribute required by the model
    data.batch = None
    
    return data

def set_labels(x):
    labels = []
    for i in range(len(x)):
        if (x[i][0] != 0): labels.append("C")
        elif (x[i][1] != 0): labels.append("N")
        elif (x[i][2] != 0): labels.append("O")
        elif (x[i][3] != 0): labels.append("S")
        elif (x[i][4] != 0): labels.append("F")
        elif (x[i][5] != 0): labels.append("Si")
        elif (x[i][6] != 0): labels.append("P")
        elif (x[i][7] != 0): labels.append("Cl")
        elif (x[i][8] != 0): labels.append("Br")
        elif (x[i][9] != 0): labels.append("Mg")
        elif (x[i][11] != 0): labels.append("Na")
        elif (x[i][12] != 0): labels.append("Ca")
        elif (x[i][13] != 0): labels.append("Fe")
        elif (x[i][14] != 0): labels.append("As")
        elif (x[i][15] != 0): labels.append("Al")
        elif (x[i][16] != 0): labels.append("I")
        elif (x[i][17] != 0): labels.append("B")
        elif (x[i][18] != 0): labels.append("V")
        elif (x[i][19] != 0): labels.append("K")
        elif (x[i][20] != 0): labels.append("Tl")
        elif (x[i][21] != 0): labels.append("Yb")
        elif (x[i][22] != 0): labels.append("Sb")
        elif (x[i][23] != 0): labels.append("Sn")
        elif (x[i][24] != 0): labels.append("Ag")
        elif (x[i][25] != 0): labels.append("Pd")
        elif (x[i][26] != 0): labels.append("Co")
        elif (x[i][27] != 0): labels.append("Se")
        elif (x[i][28] != 0): labels.append("Ti")
        elif (x[i][29] != 0): labels.append("Zn")
        elif (x[i][30] != 0): labels.append("H")
        elif (x[i][31] != 0): labels.append("Li")
        elif (x[i][32] != 0): labels.append("Ge")
        elif (x[i][33] != 0): labels.append("Cu")
        elif (x[i][34] != 0): labels.append("Au")
        elif (x[i][35] != 0): labels.append("Ni")
        elif (x[i][36] != 0): labels.append("Cd")
        elif (x[i][37] != 0): labels.append("In")
        elif (x[i][38] != 0): labels.append("Mn")
        elif (x[i][39] != 0): labels.append("Zr")
        elif (x[i][40] != 0): labels.append("Cr")
        elif (x[i][40] != 0): labels.append("Pt")
        elif (x[i][40] != 0): labels.append("Hg")
        elif (x[i][40] != 0): labels.append("Pb")
        else: labels.append("X")
    
    label_dict = {i: value for i, value in enumerate(labels)}
    return label_dict

def generate_graph_visualization(data, prediction):
    """Generate and save a visualization of the molecule graph"""
    
    edges = []
    for i in range(len(data.edge_index[0])):
        edges.append([data.edge_index[0][i].item(), data.edge_index[1][i].item()])
    g = nx.Graph(edges)

    node_mask = prediction.node_mask.squeeze().tolist()
    node_mask_values = {i: value for i, value in enumerate(node_mask)}
    nx.set_node_attributes(g, node_mask_values, 'value')

    edge_values = prediction.edge_mask.numpy()
    for i, (u, v) in enumerate(g.edges()):
        g[u][v]['weight'] = edge_values[i]

    labels = set_labels(data.x)
    node_values = np.array([data['value'] for _, data in g.nodes(data=True)])

    node_norm = plt.Normalize(vmin=0, vmax=1)
    edge_norm = plt.Normalize(vmin=0, vmax=1)

    node_cmap = plt.cm.Blues
    edge_cmap = plt.cm.Reds

    node_colors = [node_cmap(node_norm(value)) for value in node_values]
    edge_colors = [edge_cmap(edge_norm(weight)) for weight in edge_values]

    pos = nx.spring_layout(g)

    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            node_size=150, font_color='black', ax=ax, width=3)

    # Convert image to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300)
    plt.close(fig)
    img_buffer.seek(0)
    base64_image = base64.b64encode(img_buffer.read()).decode('utf-8')

    return base64_image

model_config = ModelConfig(
    mode='regression',
    task_level='graph',
    return_type='raw',
)

def predict_affinity(compound_smiles, target_sequence, generate_graph=False):
    model, device = load_model()
    explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=model_config,
        )

    try:
        data = process_input(compound_smiles, target_sequence)
        data = data.to(device)
        
        with torch.no_grad():
            prediction = model(data.x, data.edge_index, data).item()
        
        graph_encoding = None
        if generate_graph:
            try:
                explanation = explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    data = data
                )

                # generate base64 image using your visualization logic
                graph_encoding = generate_graph_visualization(data, explanation)
            except Exception as ex:
                graph_encoding = None  # Optionally keep this null if explanation fails

        
        return prediction, graph_encoding
    
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")
