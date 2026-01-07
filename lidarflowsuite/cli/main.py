import typer
import torch
from lidarflowsuite.utils.config import load_config
from lidarflowsuite.models.model import SceneFlowModel
from lidarflowsuite.training.trainer import Trainer
from lidarflowsuite.models.losses import SceneFlowLoss
from torch.utils.data import DataLoader
from lidarflowsuite.data.kitti import KITTIDataset
import uvicorn

app = typer.Typer()

@app.command()
def train(config: str = "configs/default_train.yaml"):
    """Starts the training process."""
    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SceneFlowModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))
    loss_fn = SceneFlowLoss()
    
    # Placeholder datasets
    train_ds = KITTIDataset(root_dir=cfg.get('data_root'))
    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 4), shuffle=True)
    
    trainer = Trainer(model, optimizer, loss_fn, train_loader, None, device, cfg)
    trainer.train(num_epochs=cfg.get('epochs', 10))

@app.command()
def api(port: int = 8000):
    """Starts the FastAPI inference server."""
    from lidarflowsuite.api.app import app as fastapi_app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

@app.command()
def dashboard():
    """Starts the Streamlit visualization dashboard."""
    import subprocess
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "visualization", "dashboard.py")
    subprocess.run(["streamlit", "run", dashboard_path])

if __name__ == "__main__":
    app()
