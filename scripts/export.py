import torch
import onnx
from lidarflowsuite.models.model import SceneFlowModel

def export_to_onnx(checkpoint_path, output_path="model.onnx"):
    model = SceneFlowModel()
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Dummy inputs for shape trace
    pc1 = torch.randn(1, 2048, 3)
    pc2 = torch.randn(1, 2048, 3)
    
    torch.onnx.export(
        model, 
        (pc1, pc2), 
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['pc1', 'pc2'],
        output_names=['flow'],
        dynamic_axes={'pc1': {0: 'batch_size', 1: 'num_points'},
                      'pc2': {0: 'batch_size', 1: 'num_points'},
                      'flow': {0: 'batch_size', 1: 'num_points'}}
    )
    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="lidarflow.onnx")
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output)
