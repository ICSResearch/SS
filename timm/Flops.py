import torch
from models import create_model
import torch.nn.functional as F
from fvcore.nn.parameter_count import parameter_count
from fvcore.nn.flop_count import flop_count


model = create_model('dnet_12_224')

# input = torch.zeros(size=[1, 3, 224, 224])
# input = torch.randn(size=(1, 3, 224, 224))
input1 = torch.randn(size=(1, 3, 224, 224))

p1 = parameter_count(model)

print(f'original: {sum(p1.values())}')

f1, _ = flop_count(model, input1)


print(f'original: {sum(f1.values())}')




