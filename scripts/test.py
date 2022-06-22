import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

print(dir(model))

model.train()
