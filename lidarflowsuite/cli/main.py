import typer
import torch
from lidarflowsuite.utils.config import load_config
from lidarflowsuite.models.model import SceneFlowModel
from lidarflowsuite.training.trainer import Trainer
from lidarflowsuite.models.losses import SceneFlowLoss
from torch.utils.data import DataLoader
from lidarflowsuite.data.kitti import KITTIDataset
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
import os

console = Console()
app = typer.Typer(help="LidarFlowSuite: Professional 3D Scene Flow Solutions")

def show_banner():
    banner_text = """
    [bold cyan]
    __    _     __           ______               _____       _ __     
   / /   (_)___/ /___ ______/ ____/___ _      __ / ___/__  __(_) /____ 
  / /   / / __  / __ `/ ___/ /_  / __ \ | /| / / \__ \/ / / / / __/ _ \\
 / /___/ / /_/ / /_/ / /  / __/ / /_/ / |/ |/ / ___/ / /_/ / / /_/  __/
/_____/_/\__,_/\__,_/_/  /_/    \____/|__/|__/ /____/\__,_/_/\__/\___/ 
    [/bold cyan]
    [italic magenta]Minimum Viable Product (MVP) - Self-Supervised 3D Motion Estimation[/italic magenta]
    """
    console.print(Panel(banner_text, subtitle="v0.1.0", border_style="blue"))

@app.command()
def train(config: str = typer.Option("configs/default_train.yaml", help="Path to the config file")):
    """Starts the self-supervised training process with style."""
    show_banner()
    with console.status("[bold green]Initializing training engine...") as status:
        try:
            cfg = load_config(config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rprint(f"[bold cyan]Device:[/bold cyan] {device}")
            
            model = SceneFlowModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))
            loss_fn = SceneFlowLoss()
            
            # Placeholder datasets
            train_ds = KITTIDataset(root_dir=cfg.get('data_root'))
            train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 4), shuffle=True)
            
            status.update("[bold yellow]Dataset loaded. Starting trainer...")
            trainer = Trainer(model, optimizer, loss_fn, train_loader, None, device, cfg)
            trainer.train(num_epochs=cfg.get('epochs', 10))
            
            console.print("[bold green]Success:[/bold green] Training completed successfully!")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

@app.command()
def api(port: int = typer.Option(8000, help="Port to run the API on")):
    """Starts the FastAPI inference server with style."""
    show_banner()
    console.print(f"[bold cyan]Starting API server on port {port}...[/bold cyan]")
    from lidarflowsuite.api.app import app as fastapi_app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

@app.command()
def eval(dataset: str = "kitti", checkpoint: str = typer.Option(None, help="Path to model checkpoint")):
    """Runs the formal evaluation pipeline on a dataset."""
    show_banner()
    from lidarflowsuite.training.evaluator import Evaluator
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SceneFlowModel()
    
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        rprint(f"[bold green]Loaded checkpoint:[/bold green] {checkpoint}")
    
    # Dataset selection
    if dataset == "kitti":
        ds = KITTIDataset(root_dir="data/kitti")
    else:
        from lidarflowsuite.data.nuscenes import NuScenesDataset
        ds = NuScenesDataset(nusc_root="data/nuscenes")
        
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    evaluator = Evaluator(model, loader, device)
    evaluator.evaluate()

@app.command()
def dashboard():
    """Starts the Streamlit visualization dashboard."""
    show_banner()
    console.print("[bold cyan]Launching Streamlit Dashboard...[/bold cyan]")
    import subprocess
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "visualization", "dashboard.py")
    subprocess.run(["streamlit", "run", dashboard_path])

@app.command()
def status():
    """Shows the current status and statistics of the project."""
    show_banner()
    table = Table(title="LidarFlowSuite Status Overview")
    table.add_column("Module", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="magenta")
    
    table.add_row("Core Model", "Ready", "v0.1.0")
    table.add_row("Data Pipeline", "Ready (KITTI)", "v0.1.0")
    table.add_row("Training Engine", "Enhanced (TB)", "v0.1.0")
    table.add_row("API / CLI", "Rich UI Enabled", "v0.1.0")
    
    console.print(table)

if __name__ == "__main__":
    app()
