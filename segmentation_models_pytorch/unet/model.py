from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
import torch.nn as nn
import torch

class PPM(torch.nn.Module):
    def __init__(self,in_shape,target_out=None):
        super().__init__()
        # in_shape: shape of last features in features list
        # return feature of same shape to be merged with regular decoder
        pool_scales = (1, 2, 3, 6)
        if not target_out: target_out=in_shape
        self.ppm = []
         
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_shape, 256, kernel_size=1, bias=True), # creating 256 feats 
                nn.BatchNorm2d(256),
                nn.LeakyReLU()
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.out_conv=nn.Conv2d(in_shape + len(pool_scales) * 256, target_out, kernel_size=1, bias=True) 
    def forward(self,x):
        final_feature=x[-1]
        
        input_size = final_feature.size()
        ppm_out = [final_feature]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(final_feature),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        final_feature = self.out_conv(ppm_out)
        
        x[-1]=final_feature
#         print('out ppm')

        return(x)
    


        
class Unet(SegmentationModel):
    """ decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
    """
    
    """
    Modifications:
    encoder_name: added encoder dilated_resnetX, instead of maxpool can dilate block3,4
    Skip original image
    ppm
    undo_imagenet_norm for image for easier fg residual prediction
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16), # Decoder is NOT symettric!
        ppm=False,                                               #*** Added
        undo_imagenet_norm=False,                                 #*** Added
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
   
        
        encoder_out_channels=list(self.encoder.out_channels)
        if ppm:  # only works for resnet -> last_channel
            target_out=self.encoder.out_channels[-2] # apply ppm and reduce dimension to upper block shape
            if hasattr(self.encoder.layer4[-1],'conv3'): last_channel=self.encoder.layer4[-1].conv3.out_channels # resnet50
            else: last_channel=self.encoder.layer4[-1].conv2.out_channels # resnet34
            self.ppm=PPM(last_channel,target_out) 
            encoder_out_channels[-1]=encoder_out_channels[-2] # output of encoder is now out_channels[-2]
        else: self.ppm=None
        

        self.decoder = UnetDecoder(
            encoder_channels=tuple(encoder_out_channels),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            undo_imagenet_norm=undo_imagenet_norm,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        
        self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

        
        #change decoder channels after ppm, not allowing to change self.encoder.out_channels tuple even with list conversion
# if ppm: 
#             target_out=self.encoder.out_channels[-2] 
#             if hasattr(self.encoder.layer4[-1],'conv3'): last_channel=self.encoder.layer4[-1].conv3.out_channels
#             else: last_channel=self.encoder.layer4[-1].conv2.out_channels
#             self.ppm=PPM(last_channel,target_out) # only works for resnet
#             self.encoder.out_channels[-1]=self.encoder.out_channels[-2]
#         else: self.ppm=None