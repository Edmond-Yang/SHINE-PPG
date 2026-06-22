import os
import time
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from thop import profile
from typing import Dict, List, Optional, Tuple, Union

# Explicit imports instead of wildcard
from utils.path import pathManager
from utils.metrics import predict_heart_rate, butter_bandpass

# Initialize logger
log = logging.getLogger(__name__)


@dataclass
class ModelParams:
    mode: str = None
    load: list = None
    length: int = 320
    n_step: int = 1
    kwargs: Dict = field(default_factory=dict)
    protocol: list = None
    model_name: str = None
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, **kwargs):
        _fail_update = {}
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                _fail_update[k] = v
        if _fail_update:
            self.kwargs.update(_fail_update)

@dataclass
class ForwardParams:
    epoch: int
    step: int
    data: Dict
    kwargs: Dict = field(default_factory=dict)


class ModelTemplate(ABC):

    def __init__(self, model_params: ModelParams):

        self.model: Optional[torch.nn.Module] = None
        self.model_name = model_params.model_name.lower()

        
        self.mode = model_params.mode
        self.load = model_params.load
        self.length = model_params.length
        self.device = model_params.device
        self.n_step = model_params.n_step
        self.kwargs = model_params.kwargs
        self.protocol = " + ".join(model_params.protocol)

        self.need_output_and_export = True
        self.epoch = 0
        self.step = 0

        self.weight: Dict[str, float] = {}
        
        # Loss management
        self.loss_name: List[str] = []
        self.batch_loss: Dict[str, float] = {}
        self.epoch_loss: Dict[str, float] = {}


    # =============================================================================
    # Main Flow
    # =============================================================================
    def __call__(self, data: ForwardParams):
        self.epoch = data.epoch
        self.step = data.step

        if self.mode == 'train':
            return self._run_train(data)
        elif self.mode == 'inference':
            return self._run_inference(data)
        elif self.mode == 'output':
            return self._run_output(data)
        elif self.mode == 'efficiency':
            return self._run_efficiency(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported.")

    def _run_train(self, data: ForwardParams):
        self.preprocess()
        loss = self.train(data.data)
        self.postprocess()
        return loss

    def _run_inference(self, data: ForwardParams):
        if self._should_load_weight():
            self.load_weight()
        return self.inference(data.data)

    def _run_output(self, data: ForwardParams):
        if self._should_load_weight():
            self.load_weight()
        if self.kwargs.get('output_type', 'None').lower() == 'signal':
            return self.output_signal(data.data, kwargs=data.kwargs)
        if self.kwargs.get('output_type', 'None').lower() == 'saliency':
            return self.output_saliency(data.data, kwargs=data.kwargs)
        return self.output(data.data)

    def _run_efficiency(self, data: ForwardParams):
        if self._should_load_weight():
            self.load_weight()
        return self.efficiency(data.data)

    def _should_load_weight(self):
        # Specific models might handle weight loading differently or implicitly
        return self.model_name not in ['pos', 'chrom'] and self.step == 1

    # =============================================================================
    # Abstract Methods & Core Interface
    # =============================================================================
    @abstractmethod
    def train(self, data: Dict):
        """Perform a single training step."""
        pass

    @abstractmethod
    def inference(self, data: Dict):
        """Perform inference on the data."""
        pass

    def output(self, data: Dict):
        """
        Generate and save output.
        Not marked as abstract to allow flexibility, but raises NotImplementedError by default.
        """
        raise NotImplementedError("Output method is not implemented for this model.")

    def output_saliency(self, data: Dict, kwargs: Dict):
        raise NotImplementedError("output_saliency is not implemented for this model.")

    def get_inference_inputs(self, data: Dict):
        """
        Extract inputs for efficiency profiling.
        Should return a tensor or a tuple of tensors.
        """
        pass

    # =============================================================================
    # Training Loop Helpers
    # =============================================================================
    def preprocess(self):
        """Pre-step operations (e.g., zero_grad)."""
        if hasattr(self, 'optimizer'):
            self.optimizer.zero_grad()

    def postprocess(self):
        """
        Post-step operations: update metrics, log progress, reset batch loss, save weights.
        """
        # 1. Update epoch loss
        for name, value in self.batch_loss.items():
            self.epoch_loss[name] += value / self.n_step

        # 2. Log batch loss
        # Use log.detail for verbose training logs
        if hasattr(log, 'detail') and self.need_output_and_export:
            loss_str = self.generate_log_message(self.batch_loss)
            log.detail(f"Epoch: {self.epoch:<3d} | Step: {self.step:<5d}/{self.n_step:<5d}")
            for sentence in loss_str:
                log.detail(f"{sentence}")

        # 3. Reset batch loss
        self.reset_loss(self.batch_loss)

        # 4. End of epoch operations
        if self.step == self.n_step:
            self.save_weight()
            self._log_epoch_summary()
            self.reset_loss(self.epoch_loss)

        if hasattr(self, 'scheduler') and self.step == 1:
            self.scheduler.step()

    def _log_epoch_summary(self):

        if self.need_output_and_export:
            
            """Log the summary of the epoch (losses and weights)."""
            epoch_loss_str = self.generate_log_message(self.epoch_loss)
            weight_msg = self.generate_log_message(self.weight) if self.weight else "N/A"

            log.info(f"⎡⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺")
            log.info(f"⎮ Epoch {self.epoch} Summary")
            for sentence in epoch_loss_str:
                log.info(f"⎮ Epoch Loss : {sentence}")
            for sentence in weight_msg:
                log.info(f"⎮ Weight : {sentence}")
            log.info(f"⎣⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽⎽")

    def update(self, **kwargs):
        origin_data = dict()
        for k, v in kwargs.items():
            if v is not None:
                origin = getattr(self, k)
                setattr(self, k, v)
                origin_data[k] = origin
        return origin_data

    # =============================================================================
    # Loss & Metric Management
    # =============================================================================
    def init_loss(self):
        self.batch_loss = {}
        self.epoch_loss = {}

        for name in (self.loss_name + ['total']):
            self.batch_loss[name] = 0.
            self.epoch_loss[name] = 0.

        for name in self.loss_name:
            self.weight[name] = 1.0

    def reset_loss(self, loss_dict: Dict[str, float]):
        for k in loss_dict.keys():
            loss_dict[k] = 0.

    def calculate_loss(self):
        total = 0.
        for name in self.loss_name:
            total += self.weight[name] * self.batch_loss[name]
            # Handle 0-dim tensor vs float
            if hasattr(self.batch_loss[name], 'item'):
                self.batch_loss[name] = self.batch_loss[name].item()
        
        self.batch_loss["total"] = total.item() if hasattr(total, 'item') else total
        return total

    def generate_log_message(self, loss_dict: Dict) -> List[str]:
        """Format loss dictionary into readable strings."""
        items = []
        if 'total' in loss_dict:
            items.append(f"[{'total':<20}]: {loss_dict['total']:.4f}")

        other_items = sorted([
            f"[{name:<20}] {value:.4f}" for name, value in loss_dict.items() if name != 'total'
        ])
        return items + other_items

    # =============================================================================
    # IO & Checkpointing
    # =============================================================================
    def get_weight_path(self, protocol, component=None):
        return pathManager.get_weight_path(self.model_name, protocol, self.length, self.epoch, component)

    def load_weight(self):
        path = self.get_weight_path(self.load)
        if self.model and os.path.exists(path) and self.need_output_and_export:
            log.detail(f"Loading weights from {path}")
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        elif self.need_output_and_export:
            log.warning(f"Weight path {path} does not exist or model is None.")

    def save_weight(self):
        if self.model and self.need_output_and_export:
            path = self.get_weight_path(self.protocol)
            log.detail(f"Saving weights to {path}")
            torch.save(self.model.state_dict(), path)

    # =============================================================================
    # Tools / Efficiency / Visualization
    # =============================================================================
    def efficiency(self, data: Dict):

        # function exists
        if not hasattr(self, 'get_model_inputs'):
            log.warning("No get_model_inputs function found. Skipping efficiency profiling.")
            return

        sample_input = self.get_model_inputs(data)
        inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)
        
        try:
            flops, params = profile(self.model, inputs=inputs, verbose=False)
        except Exception as e:
            log.error(f"Error profiling model: {e}")
            flops, params = 0, 0

        # Execution time
        repetitions = 10
        timings = []
        # Warmup
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        for _ in range(repetitions):
            start_time = time.time()
            _ = self.model(*inputs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            
        avg_time = sum(timings) / repetitions
        log.info(f"FLOPs: {flops/1e9:.2f} G, Params: {params/1e6:.2f}M, Average Inference Time: {(avg_time * 1000):.2f} ms")

    # =============================================================================
    # Output Signal Visualization
    # =============================================================================

    def output_signal(self, data: Dict, kwargs: Dict):
        pred = self.inference(data)
        gt = data['gt']
        out_dir = kwargs.get('out_dir', "output")
        info = data['info']
        return self._output_signal_figure(pred, gt, out_dir, info)
    

    def _signal_to_gaf(self, signal):
        phi = np.arccos(signal)
        gaf = np.cos(phi[:, None] + phi[None, :])
        return gaf

    def _output_signal_figure(self, preds, gts, out_dir, info):
        """Helper to process and save signal plots."""
        if isinstance(preds, torch.Tensor):
            pred = preds.detach().cpu().numpy()
        if isinstance(gts, torch.Tensor):
            gt = gts.detach().cpu().numpy()

        # Hardcoded filter params preserved from original
        pred = butter_bandpass(pred, 40. / 60, 250. / 60, 30)
        gt = butter_bandpass(gt, 40. / 60, 250. / 60, 30)

        hr_pred = predict_heart_rate(pred, info['fps'])
        hr_gt = predict_heart_rate(gt, info['fps'])

        B = pred.shape[0]

        for i in range(B):
            p = pred[i]
            g = gt[i]
            
            name = f"{self.epoch:04d}/{info['dataset'][i]}_{info['video'][i]}"

            dataset = info['dataset'][i]
            video = info['video'][i]
            start = info['start'][i]
            end = info['end'][i]

            save_dir = out_dir / name
            pathManager.make_dir(save_dir)

            # Normalize
            pred_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
            gt_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)

            # Signal output
            save_pred_signal_path = save_dir / f'pred-signal.txt'
            save_gt_signal_path = save_dir / f'gt-signal.txt'
            np.savetxt(save_pred_signal_path, pred_norm, fmt='%.6f', comments='', newline=' ')
            np.savetxt(save_gt_signal_path, gt_norm, fmt='%.6f', comments='', newline=' ')

            # GAF output
            pred_gaf = self._signal_to_gaf(pred_norm)
            gt_gaf = self._signal_to_gaf(gt_norm)
            save_pred_gaf_path = save_dir / f'pred-gaf.png'
            save_gt_gaf_path = save_dir / f'gt-gaf.png'
            plt.imsave(save_pred_gaf_path, pred_gaf, cmap='jet', vmin=-1.0, vmax=1.0)
            plt.imsave(save_gt_gaf_path, gt_gaf, cmap='jet', vmin=-1.0, vmax=1.0)

            # Plot
            plt.figure(figsize=(6, 2.5))
            plt.plot(gt_norm, color='#1f77b4', label=f'GT HR: {int(hr_gt[i])} bpm', linewidth=1.5, alpha=0.5)
            plt.plot(pred_norm, color='#ff7f0e', label=f'HR: {int(hr_pred[i])} bpm', linewidth=1.5)

            # Reconstruct log message safely
            log.info(f"Epoch {self.epoch}: GT HR = {hr_gt[i]:.2f} bpm, Predicted HR = {hr_pred[i]:.2f} bpm for {dataset}/{video} - [{start}:{end}]")

            plt.tight_layout()

            save_img_path = save_dir / f'signal.png'            
            plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
            plt.close()

        return {}
    

    def output_saliency(self, data: Dict, kwargs: Dict):
        
        # ensure gradients are enabled
        self.model.train()
        self.model.zero_grad()
        
        grad = self.saliency(data)
        
        