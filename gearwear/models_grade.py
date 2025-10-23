
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- ResNet-like backbone with SE ---------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid_ch = out_ch * 2
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction=8)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x), inplace=True)

class EnhancedResNet18(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._init_weights()
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks): layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        feat = self.pool(x).view(x.size(0), -1)
        return feat  # [B,512]

# --------- Axes heads & CORAL grade ---------
class HeadsAreaDepth(nn.Module):
    def __init__(self, in_dim=512, depth_bins=0):
        super().__init__()
        self.depth_bins = depth_bins
        self.area = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        if depth_bins > 0:
            self.depth = nn.Linear(in_dim, depth_bins-1)
        else:
            self.depth = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 1)
            )
    def forward(self, feat):
        out = {'area_ratio_pred': self.area(feat)}
        if self.depth_bins > 0:
            logits = self.depth(feat)
            out.update({'depth_logits': logits, 'depth_probs': torch.sigmoid(logits)})
        else:
            out.update({'depth_mm_pred': self.depth(feat)})
        return out

class GradeFromAxes_CORAL(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        assert num_classes >= 2
        self.num_classes = num_classes
        self.raw_wd  = nn.Parameter(torch.tensor(0.1))
        self.raw_wa  = nn.Parameter(torch.tensor(0.1))
        self.raw_wda = nn.Parameter(torch.tensor(0.0))
        self.b       = nn.Parameter(torch.tensor(0.0))
        self.raw_th  = nn.Parameter(torch.linspace(-1.0, 1.0, num_classes-1))
    def forward(self, depth_val, area_ratio):
        wd  = F.softplus(self.raw_wd)
        wa  = F.softplus(self.raw_wa)
        wda = F.softplus(self.raw_wda)
        score = wd*depth_val + wa*area_ratio + wda*(depth_val*area_ratio) + self.b  # [B,1]
        theta, _ = torch.sort(self.raw_th)                                          # [K-1]
        logits = score - theta.view(1, -1)
        probs  = torch.sigmoid(logits)
        return {'logits': logits, 'probs': probs, 'score': score, 'theta': theta, 'weights': (wd,wa,wda,self.b)}

def coral_targets(y, K):
    thr = torch.arange(K-1, device=y.device).view(1,-1).expand(y.size(0), -1)
    return (y.view(-1,1) > thr).float()

def coral_loss(logits, y, K):
    tgt = coral_targets(y, K)
    return F.binary_cross_entropy_with_logits(logits, tgt, reduction='mean')

class GradeModel(nn.Module):
    """
    Input: 4ch image (RGB + aux channel), where aux can be predicted mask or heatmap.
    Output: area_ratio (pred), depth (pred/ordinal), grade (from axes via CORAL).
    """
    def __init__(self, in_channels=4, num_classes=5, depth_bins=0):
        super().__init__()
        self.num_classes = num_classes
        self.depth_bins = depth_bins
        self.backbone = EnhancedResNet18(in_channels=in_channels)
        self.heads = HeadsAreaDepth(512, depth_bins=depth_bins)
        self.grade_head = GradeFromAxes_CORAL(num_classes)

    def forward(self, x, area_ratio_input=None, depth_input=None):
        feat = self.backbone(x)
        out = self.heads(feat)
        # choose depth/area for grade head
        if depth_input is not None:
            depth_val = depth_input
        else:
            if 'depth_mm_pred' in out: depth_val = out['depth_mm_pred']
            else: depth_val = (out['depth_probs']>0.5).sum(dim=1, keepdim=True).float()
        if area_ratio_input is not None:
            area_val = area_ratio_input
        else:
            area_val = out['area_ratio_pred']
        out['grade'] = self.grade_head(depth_val, area_val)
        return out
