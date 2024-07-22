import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from transformers import *


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vmodel = ViTModel.from_pretrained('google/vit-base-patch16-224')


class VisionP:
    def __init__(self, vmodel, lm,processor=None):
        self.vmodel=vmodel
        self.process=processor
        self.lm=lm

    def features(self, img):
        return self.vmodel(img)[0]
    
    def get_embedding(self,features, embeddings):
        #Image/n/Text
        
        return torch.cat((self.lm.transformer.vision_layer(features[:, :1, :]),self.lm.transformer.wte(embeddings)), dim=1)
 

image="C:\\Users\\Emmanuel\\Desktop\\transformers-main\\transformers-main\\src\\transformers\\models\\gpt2\\drake.jpg"
image=Image.open(image)
image=processor(image, return_tensors="pt")['pixel_values']


g2=GPT2LMHeadModel.from_pretrained('gpt2')
tok=AutoTokenizer.from_pretrained("gpt2")

for name, i in g2.named_parameters():
    if "vision_layer" in name:
        i.requires_grad=True
    else:
        i.requires_grad=False

    print(name, i.requires_grad)

print(g2.transformer.vision_layer)


tokens=tok.encode("|start|hello we", return_tensors="pt")
process=VisionP(vmodel=vmodel, lm=g2)
feat=process.features(image)
#print(feat.shape)
input_embeddings=process.get_embedding(feat, embeddings=tokens)

print(input_embeddings.shape)

#outs=g2(input_embeddings, labels=tokens)
print(tokens.shape)



labels=torch.cat([torch.full((1, 1), -100, dtype=torch.long), tokens.clone()], dim=1)

r1=g2(inputs_embeds=input_embeddings, labels=labels)
print(r1.loss)
r1.loss.backward()
from torch.optim import AdamW
optim=AdamW(g2.parameters(), lr=1e-5)
optim.step()
optim.zero_grad()


    


