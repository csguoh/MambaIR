from thop import profile
import os
from thop import clever_format
from model_zoo.swinIR import buildSwinIR
from model_zoo.rcan import buildRCAN
from model_zoo.edsr import buildEDSR
from model_zoo.mambair import buildMambaIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H=640
    W=640
    scale=4
    # NOTE: the default returned model is classic SRx4 model
    init_model = buildMambaIR(upscale=scale).to(device)
    dummy_input = torch.randn(1, 3, H//scale,W//scale).to(device)
    flops, params = profile(init_model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops)
    print(params)

