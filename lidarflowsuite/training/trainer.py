from torch.utils.tensorboard import SummaryWriter
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
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for pc1, pc2 in pbar:
            pc1, pc2 = pc1.to(self.device), pc2.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (simplified)
            # In real case, we'd need features for pc2 to warp
            feat1 = self.model.backbone(pc1.transpose(1, 2)).transpose(1, 2)
            feat2 = self.model.backbone(pc2.transpose(1, 2)).transpose(1, 2)
            
            # Simple warping approximation for self-supervised
            # Ideally use a proper correlation layer
            pred_flow = self.model.flow_head(feat1, pc1, feat1) 
            
            loss = self.loss_fn(pc1, pc2, pred_flow)
            
            # Anomaly check
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Anomaly detected! Loss is {loss}. Skipping batch.")
                continue
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            step = (epoch - 1) * len(self.train_loader) + pbar.n
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
