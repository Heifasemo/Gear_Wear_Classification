
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Minimal Grad-CAM for a model with a conv "layer4" as last conv (EnhancedResNet18).
    """
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.target_layer = dict([*model.backbone.named_children()])[target_layer_name]
        self.activations = None
        self.gradients = None
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def __del__(self):
        try:
            self.fwd_handle.remove()
            self.bwd_handle.remove()
        except Exception:
            pass

    def generate(self, input_tensor, score):
        """
        input_tensor: [1,C,H,W]
        score: a scalar tensor from which to backprop (e.g., grade score mean)
        returns heatmap in [0,1] of size [H,W]
        """
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        # global-average the gradients
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,h,w]
        cam = F.relu(cam)
        # normalize to [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        # upsample to input size
        cam_up = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        return cam_up.squeeze(0).squeeze(0).detach().cpu()
