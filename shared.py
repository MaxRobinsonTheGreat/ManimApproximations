from manim import *
import torch

def create_weights_matricies(net):
    matricies = []
    bias, weight = None, None
    for name, param in net.named_parameters():
        if name.find('bias') != -1:
            bias = param.data
        elif name.find('weight') != -1:
            weight = param.data
        if bias is not None and weight is not None:
            # print(bias, weight)
            bias = bias.unsqueeze(1)
            matrix = torch.cat((bias, weight), dim=1).numpy()
            # round to 3 decimal places
            matrix = np.round(matrix, 3)
            bias, weight = None, None
            # display matrix as mojbect
            matrix = Matrix(matrix, h_buff=2)
            if len(matricies) > 0:
                matrix.next_to(matricies[-1], RIGHT)
            matricies.append(matrix)
    group = VGroup(*matricies).center().shift(DOWN)
    return group