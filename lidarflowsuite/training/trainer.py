from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from lidarflowsuite.utils.logger import logger
from lidarflowsuite.utils.metrics import compute_epe

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.log_dir = config.get('log_dir', 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as progress:
            task = progress.add_task(f"[yellow]Epoch {epoch}[/yellow]", total=len(self.train_loader))
            
            for pc1, pc2 in self.train_loader:
                pc1, pc2 = pc1.to(self.device), pc2.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass (multi-scale list)
                pred_flows = self.model(pc1, pc2)
                pred_flow = pred_flows[0] # Take full scale for loss computation
                
                loss = self.loss_fn(pc1, pc2, pred_flow, model=self.model)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Anomaly detected! Loss is {loss}. Skipping batch.")
                    continue
                    
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress.update(task, advance=1, description=f"[yellow]Epoch {epoch}[/yellow] [cyan]Loss: {loss.item():.4f}[/cyan]")
                
                # Log to TensorBoard
                step = (epoch - 1) * len(self.train_loader) + progress.tasks[0].completed
                self.writer.add_scalar('Loss/train', loss.item(), step)
            
        return total_loss / len(self.train_loader)

    def save_checkpoint(self, epoch, name="latest.pth"):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} finished. Average Loss: {loss:.4f}")
            self.writer.add_scalar('Loss/epoch', loss, epoch)
            self.save_checkpoint(epoch)
        self.writer.close()
