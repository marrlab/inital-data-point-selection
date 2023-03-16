
import torch
from PIL import Image

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def create_remap_values(remapping):
    def fun(x):
        index = torch.bucketize(x.ravel(), remapping[0])
        return remapping[1][index].reshape(x.shape).int()

    return fun

def get_output_dim(model, preprocess):
    model.eval()

    # passing dummy data through the model to get the output dimension
    dummy_image = Image.new('RGB', (1000, 1000))
    dummy_tensor = preprocess(dummy_image)
    dummy_batch = dummy_tensor[None, :, :, :] 

    if torch.cuda.is_available():
        dummy_batch = dummy_batch.to('cuda')

    assert len(dummy_batch.shape) == 4 and \
        dummy_batch.shape[0] == 1 and \
        dummy_batch.shape[1] == 3 and \
        dummy_batch.shape[2] == dummy_batch.shape[3]

    dummy_output = model(dummy_batch)

    assert len(dummy_output.shape) == 2 and dummy_output.shape[0] == 1

    return dummy_output.shape[1]