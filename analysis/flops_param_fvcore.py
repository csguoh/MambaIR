import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from analysis.model_zoo.mambair import buildMambaIR
from analysis.model_zoo.mambairv2 import buildMambaIRv2Small, buildMambaIRv2Large,buildMambaIRv2Base
from model_zoo.mambair import buildMambaIR_light
from model_zoo.swinIR import buildSwinIR_light
from model_zoo.hat import buildHAT
from model_zoo.dat import buildDAT
from model_zoo.cat import buildCAT
from model_zoo.mambairv2 import buildMambaIRv2_light
from analysis.utils_fvcore import FLOPs
fvcore_flop_count = FLOPs.fvcore_flop_count



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H=720
    W=1280
    scale=4
    init_model = buildMambaIRv2Small(upscale=scale).to(device)
    with torch.no_grad():
        FLOPs.fvcore_flop_count(init_model, input_shape=(3, H//scale,W//scale))


        