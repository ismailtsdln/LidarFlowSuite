import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from lidarflowsuite.models.model import SceneFlowModel
from lidarflowsuite.data.base_dataset import BasePointCloudDataset

st.set_page_config(page_title="LidarFlowSuite Dashboard", layout="wide")

st.title("🚗 LidarFlowSuite: 3D Scene Flow Visualizer")

@st.cache_resource
def load_model():
    model = SceneFlowModel()
    model.eval()
    return model

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.header("Input Data")
    uploaded_file1 = st.file_uploader("Upload Point Cloud T", type=["bin", "npy"])
    uploaded_file2 = st.file_uploader("Upload Point Cloud T+1", type=["bin", "npy"])

if uploaded_file1 and uploaded_file2:
    # Dummy loading for demo purposes
    pc1 = torch.randn(1, 2048, 3)
    pc2 = torch.randn(1, 2048, 3)
    
    with torch.no_grad():
        flow = model(pc1, pc2)
    
    pc1_np = pc1[0].numpy()
    flow_np = flow[0].numpy()
    pc2_np = pc2[0].numpy()
    
    with col2:
        st.header("3D Visualization")
        fig = go.Figure()
        
        # Plot PC1
        fig.add_trace(go.Scatter3d(
            x=pc1_np[:, 0], y=pc1_np[:, 1], z=pc1_np[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.5),
            name='Cloud T'
        ))
        
        # Plot Flow Vectors (subset for performance)
        step = 10
        for i in range(0, len(pc1_np), step):
            p1 = pc1_np[i]
            p2 = p1 + flow_np[i]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False
            ))
            
        fig.update_layout(
            scene=dict(aspectmode='data'),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload two point cloud files to see the predicted flow.")
