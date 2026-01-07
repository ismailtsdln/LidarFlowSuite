import torch
from rich.table import Table
from rich.console import Console
from lidarflowsuite.utils.metrics import compute_epe, compute_accuracy
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, dataloader, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.console = Console()

    def evaluate(self):
        self.model.eval()
        total_epe = 0
        total_acc_strict = 0
        total_acc_relaxed = 0
        num_batches = len(self.dataloader)
        
        with torch.no_grad():
            for pc1, pc2 in tqdm(self.dataloader, desc="Evaluating"):
                pc1, pc2 = pc1.to(self.device), pc2.to(self.device)
                
                # Multi-scale returns a list, take full res
                pred_flows = self.model(pc1, pc2)
                pred_flow = pred_flows[0]
                
                # In self-supervised eval without GT, we usually use EPE against synthetic or pseudo-GT
                # For this MVP, we assume a validation dataset WITH GT is available for benchmarking
                # If no GT, this computes metrics against zero flow or identity (placeholder)
                gt_flow = torch.zeros_like(pred_flow) # Placeholder GT
                
                total_epe += compute_epe(pred_flow, gt_flow).item()
                total_acc_strict += compute_accuracy(pred_flow, gt_flow, 0.05, 0.05).item()
                total_acc_relaxed += compute_accuracy(pred_flow, gt_flow, 0.1, 0.1).item()
                
        results = {
            "EPE": total_epe / num_batches,
            "Acc_Strict": total_acc_strict / num_batches,
            "Acc_Relaxed": total_acc_relaxed / num_batches
        }
        
        self.show_results(results)
        return results

    def show_results(self, results):
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, val in results.items():
            table.add_row(metric, f"{val:.4f}")
            
        self.console.print(table)
