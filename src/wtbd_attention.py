"""YOLOv8 adapters for pinned, unmodified ECA, CBAM and Coordinate Attention.

The original C2f parameter names and YAML layer indices are preserved. Each
enabled block applies one canonical attention module to the complete C2f output.
Existing ESB/LSK classes and checkpoints remain in wtbd_modules unchanged.
"""
from torch import nn
from ultralytics.nn.modules import C2f
from wtbd_attention_vendor.eca import eca_layer
from wtbd_attention_vendor.cbam import CBAM
from wtbd_attention_vendor.ca import CoordAtt

ATTENTION_TYPES = {'eca': eca_layer, 'cbam': CBAM, 'ca': CoordAtt}

def make_attention(name, channels):
    if name == 'none':
        return nn.Identity()
    if name == 'eca':
        return eca_layer(channels, k_size=3)
    if name == 'cbam':
        return CBAM(channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
    if name == 'ca':
        return CoordAtt(channels, channels, reduction=32)
    raise ValueError(f'Unknown attention: {name}')

class C2fAttention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, attention='none'):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.attention_name = attention
        self.feature_block = make_attention(attention, c2)

    def forward(self, x):
        return self.feature_block(super().forward(x))

    def forward_split(self, x):
        return self.feature_block(super().forward_split(x))

def register_attention_modules():
    import ultralytics.nn.tasks as tasks
    from wtbd_modules import register_wtbd_modules
    register_wtbd_modules()
    tasks.C2f = C2fAttention
