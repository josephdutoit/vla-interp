import os
import sys

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import pickle
import numpy as np
from enum import Enum

from io import BytesIO
import base64
from PIL import Image, ImageDraw

from transformers import AutoTokenizer
from qwen_vl_utils import process_vision_info

from rv_train.dataset import LiberoDataset
from rv_train.model import load_processor

class FeatureClass(Enum):
    VISION = 0
    LANGUAGE = 1
    ACTION = 2
    MULTIMODAL = 3

# 1. Load data
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "data/data/discovered_features/layer_11/classified_features.pkl")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data/data/discovered_features/layer_11/global_features.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "logs/runs/vla0-trl-test")

processor = load_processor(MODEL_PATH, img_size=224, num_cams=2, tile_images=True)
dataset = LiberoDataset(repo_id="HuggingFaceVLA/libero", img_size=224)

with open(EMBEDDINGS_PATH, "rb") as f:
    umap_data = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    global_results = pickle.load(f)

# Create the dataframe
# df = pd.DataFrame({
#     'x': umap_data['embedding'][:, 0],
#     'y': umap_data['embedding'][:, 1],
#     'f_id': umap_data['f_ids'],
#     'max_act': umap_data['max_acts'],
#     'feature_class': umap_data['feature_class']
# })

df = pd.DataFrame({
    'x': umap_data['x'],
    'y': umap_data['y'],
    'f_id': umap_data['f_id'],
    'max_act': umap_data['max_act'],
    'feature_class': umap_data['feature_class']
})

df['log_max_act'] = np.log1p(df['max_act'] + 1e-6)  # Log scale for better color distribution

app = dash.Dash(__name__)

print(df['feature_class'].value_counts())

# Map feature_class to names and colors
feature_class_names = {
    'VISION': 'Vision',
    'ACTION': 'Action',
    'LANGUAGE': 'Language',
    'MULTIMODAL': 'Multimodal',
}
feature_class_colors = {
    'VISION': '#1f77b4',  # blue
    'ACTION': '#ff7f0e',  # orange
    'LANGUAGE': '#2ca02c',  # green
    'MULTIMODAL': '#d62728',  # red
}


# If feature_class is float, convert to int for mapping
df['feature_class_name'] = df['feature_class'].map(feature_class_names)
df['feature_class_color'] = df['feature_class'].map(feature_class_colors)

# Create a scatter trace for each class for legend and color

scatter_traces = []
for class_id, class_name in feature_class_names.items():
    class_mask = df['feature_class'] == class_id
    print(f"Class {class_name}: {class_mask.sum()} features")
    if class_mask.sum() == 0:
        continue
    scatter_traces.append(go.Scattergl(
        x=df.loc[class_mask, 'x'],
        y=df.loc[class_mask, 'y'],
        mode='markers',
        marker=dict(size=5, color=feature_class_colors[class_id]),
        name=class_name,
        hovertext=[f"Feature {int(fid)}" for fid in df.loc[class_mask, 'f_id']],
        customdata=df.loc[class_mask, 'f_id'],
        legendgroup=class_name
    ))

app.layout = html.Div(style={'backgroundColor': '#111', 'color': 'white', 'padding': '20px'}, children=[
    html.H1("VLA SAE Feature Explorer", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Jump to Feature Index: "),
        dcc.Input(
            id='feature-search',
            type='number',
            placeholder='e.g. 4302',
            style={'backgroundColor': '#333', 'color': 'white', 'border': '1px solid #555', 'marginLeft': '10px'}
        ),
        html.Button('Search', id='search-btn', n_clicks=0, style={'marginLeft': '5px'})
    ], style={'marginBottom': '20px', 'textAlign': 'center'}),

    html.Div([
        html.Div([
            dcc.Graph(
                id='umap-plot',
                figure=go.Figure(
                    data=scatter_traces,
                    layout=go.Layout(
                        title="UMAP Semantic Map",
                        template="plotly_dark",
                        height=750,
                        clickmode='event+select',
                        legend=dict(title='Feature Class', bgcolor='#222', font=dict(color='white'))
                    )
                )
            )
        ], style={'width': '60%', 'display': 'inline-block'}),

        html.Div([
            html.H3(id='feature-title', children="Click a dot to explore..."),
            html.Div(id='image-container')
        ], style={'width': '35%', 'float': 'right', 'padding': '20px', 'backgroundColor': '#222', 'borderRadius': '10px'})
    ])
])


@app.callback(
    [Output('feature-title', 'children'), Output('image-container', 'children')],
    [Input('umap-plot', 'clickData'),
     Input('search-btn', 'n_clicks')],
    [dash.State('feature-search', 'value')]
)
def update_view(clickData, n_clicks, search_value):
    # Determine which input triggered the callback
    triggered_id = ctx.triggered_id
    f_id = None


    if triggered_id == 'umap-plot' and clickData:
        # Use pointNumber to get the correct row index in the filtered DataFrame for the trace
        point = clickData['points'][0]
        trace_idx = point['curveNumber']
        point_number = point['pointNumber']
        # Get the mask for the trace
        class_id = list(feature_class_names.keys())[trace_idx]
        class_mask = df['feature_class'] == class_id
        class_df = df[class_mask].reset_index(drop=True)
        f_id = int(class_df.iloc[point_number]['f_id'])

    elif triggered_id == 'search-btn' and search_value is not None:
        # Get ID from search box
        f_id = int(search_value)

    if f_id is None:
        return "Select or Search for a feature...", []

    # Check if the feature exists in your global results
    if f_id not in global_results:
        return f"Feature {f_id} not found", [html.P("This feature might be dead (no activations).")]

    try:
        examples = global_results.get(f_id, [])

        if not examples:
            return f"Feature #{f_id}", [html.P("No data.")]

        cards = []
        for ex in examples:
            # --- 1. Get Image ---
            sample = dataset[ex['sample_idx']]
            images = sample['images'] # PIL Image

            # --- 2. Decode Token & Process Spatial Info ---
            messages = sample['messages']

            # Inject images into messages for process_vision_info
            img_idx = 0
            for msg in messages:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content["type"] == "image":
                            content["image"] = images[img_idx]
                            img_idx += 1

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
            input_ids = inputs['input_ids'][0]

            # Special Token IDs for Qwen2.5-VL
            IMAGE_PAD_ID = 151655

            token_idx = ex['token_idx']
            activated_token_id = input_ids[token_idx].item()

            display_image = images[0].copy()
            is_vision_token = (activated_token_id == IMAGE_PAD_ID)
            if is_vision_token and "image_grid_thw" in inputs:
                # Calculate relative position among vision tokens
                # This assumes we are looking at the first image (index 0)
                # Find the start of the vision tokens for this image
                vision_start_indices = (input_ids == 151652).nonzero(as_tuple=True)[0]
                if len(vision_start_indices) > 0:
                    v_start = vision_start_indices[0].item()
                    # Relative index within the <|image_pad|> sequence
                    # We subtract v_start + 1 because <|vision_start|> is at v_start
                    rel_idx = token_idx - (v_start + 1)

                    grid_thw = inputs["image_grid_thw"][0] # [T, H, W]
                    # print("Grid THW:", grid_thw)
                    # print("Image shape: ", display_image.size)
                    h_grid, w_grid = grid_thw[1].item() // 2, grid_thw[2].item() // 2

                    row_idx = rel_idx // w_grid
                    col_idx = rel_idx % w_grid

                    # Highlight the patch on the image
                    draw = ImageDraw.Draw(display_image, "RGBA")
                    patch_h = display_image.height / h_grid
                    patch_w = display_image.width / w_grid

                    left = col_idx * patch_w
                    top = row_idx * patch_h
                    right = left + patch_w
                    bottom = top + patch_h

                    # Draw a semi-transparent red box
                    draw.rectangle([left, top, right, bottom], fill=(255, 0, 0, 128), outline=(255, 0, 0, 255))
                    vision_coord_text = f" (Grid: {row_idx}, {col_idx})"
                else:
                    vision_coord_text = ""
            else:
                vision_coord_text = ""

            # Convert Display Image to Base64
            buf = BytesIO()
            display_image.save(buf, format="JPEG")
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            try:
                # Identify the range of tokens to display (5 left, 5 right)
                start_idx = max(0, token_idx - 5)
                end_idx = min(len(input_ids), token_idx + 6)

                context_elements = []
                for i in range(start_idx, end_idx):
                    token_id = input_ids[i].item()
                    decoded = processor.tokenizer.decode([token_id])

                    if i == token_idx:
                        # Highlight activated token
                        label = decoded
                        if is_vision_token:
                            label += vision_coord_text

                        context_elements.append(html.Span(label, style={
                            'color': '#ff9900', 
                            'fontWeight': 'bold', 
                            'backgroundColor': '#443300', 
                            'padding': '0 2px', 
                            'borderRadius': '3px',
                            'border': '1px solid #ff9900'
                        }))
                    else:
                        context_elements.append(html.Span(decoded, style={'color': '#bbb'}))

                decoded_context = context_elements
            except Exception:
                decoded_context = [html.Span("[Error Decoding]", style={'color': 'red'})]

            # --- 3. Create Visual Card ---
            cards.append(html.Div(style={
                'border': '1px solid #444', 
                'borderRadius': '8px', 
                'margin-bottom': '15px', 
                'padding': '10px',
                'backgroundColor': '#2a2a2a'
            }, children=[
                html.Img(src=f"data:image/jpeg;base64,{img_base64}", style={'width': '100%', 'borderRadius': '4px'}),
                html.P(f"Activation: {ex['value']:.4f}", style={'color': '#00ff00', 'fontWeight': 'bold', 'marginTop': '5px'}),
                html.P([html.B("Token Context: ")] + decoded_context),
                html.P(f"Instruction: {ex['instruction']}", style={'fontSize': '12px', 'marginTop': '5px'})
            ]))

        title = f"Exploring Feature #{f_id}"
        return title, cards

    except Exception as e:
        import traceback
        return f"Error: {str(e)}", [html.Pre(traceback.format_exc())]

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8050)