import matgl
import tensorflow as tf
import torch
from chgnet.model import CHGNet
from megnet.utils.models import load_model as megnet_load_model

if __name__ == '__main__':

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch_device)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#     tf_device = tf.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
    # chgnet_model = CHGNet.load().to(torch_device)
    # matgl.clear_cache()
    # band_gap_calculator = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
    # shear_modulus_calculator = megnet_load_model("logG_MP_2018")
