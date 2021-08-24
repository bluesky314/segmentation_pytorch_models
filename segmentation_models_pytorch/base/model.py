import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
            
    def freeze_encoder(self,freeze_ppm=False):
        if not freeze_ppm: print(' !!!    Freezeing Encoder NOT PPM !!!')
        else: print(' !!!    Freezeing Encoder  And PPM !!!')
        for child in self.encoder.children():
            for param in child.parameters():
                param.requires_grad = False
        if self.ppm and freeze_ppm:
            for child in self.ppm.children():
                for param in child.parameters():
                    param.requires_grad = False
    def unfreeze_encoder(self):
        print(' !!!    UnFreezeing Encoder And PPM !!!')
        
        for child in self.encoder.children():
            for param in child.parameters():
                param.requires_grad = True
        if self.ppm:
    
            for child in self.ppm.children():
                for param in child.parameters():
                    param.requires_grad = True
                
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        
        features = self.encoder(x)
        input_image=features[0]
    
        if self.ppm: features=self.ppm(features)
        
        decoder_output = self.decoder(*features)
                
        
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks, decoder_output,features

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
