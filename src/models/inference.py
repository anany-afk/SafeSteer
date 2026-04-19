import torch
import numpy as np
import logging
import os

log = logging.getLogger(__name__)

class InferenceEngine:
    """
    Unified wrapper for model inference.
    Supports PyTorch (default) and prepared for ONNX/TorchScript.
    """

    def __init__(self, model_path, cfg):
        self.device = cfg["device"]
        self.cfg = cfg
        self.model = None
        self.engine_type = "PyTorch"
        
        self.load_model(model_path)

    def load_model(self, model_path):
        """Loads the model based on extension and availability."""
        try:
            if model_path.endswith(".pth"):
                # Load PyTorch model
                from src.models.architecture import DrowsinessNet
                
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model = DrowsinessNet(
                    feat_dim    = checkpoint["feat_dim"],
                    seq_len     = self.cfg["sequence_len"],
                    cnn_channels= self.cfg["cnn_channels"],
                    lstm_hidden = self.cfg["lstm_hidden"],
                    lstm_layers = self.cfg["lstm_layers"],
                    fc_hidden   = self.cfg["fc_hidden"],
                    dropout     = 0.0,
                ).to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                log.info(f"Loaded Native PyTorch model from {model_path}")
            
            elif model_path.endswith(".onnx"):
                # TODO: Implement ONNX Runtime loader if requested
                log.warning("ONNX support requested but not yet implemented. Falling back to metrics.")
                self.engine_type = "ONNX"
            
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
                
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise

    def infer(self, sequence):
        """
        Performs inference on a preprocessed sequence.
        sequence: (seq_len, feat_dim) numpy array
        Returns: probability [0, 1]
        """
        if self.model is None:
            return 0.0
            
        try:
            # Prepare tensor
            X_t = torch.tensor(sequence[None], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(X_t)
                prob = torch.softmax(logits, dim=1)[0, 1]
                return float(prob.cpu().numpy())
        except Exception as e:
            log.error(f"Inference error: {e}")
            return 0.0

    def to_torchscript(self, save_path):
        """Exports the current model to TorchScript for faster CPU inference."""
        if self.model is None or not isinstance(self.model, torch.nn.Module):
            return
        
        dummy_input = torch.randn(1, self.cfg["sequence_len"], self.cfg["landmark_features"]).to(self.device)
        traced_script = torch.jit.trace(self.model, dummy_input)
        traced_script.save(save_path)
        log.info(f"Model exported to TorchScript: {save_path}")
