import torch
from torch import nn
import numpy as np
from manim import *
import random
import copy
import math
from models import SimpleNN, SkipConn, Fourier, SimpleTaylorNN, TaylorNN, RecurrentNN, AdaptiveExpertNN
from kan import KANLinear

# Use local search to evolve the network
def EvolveCurve(scene,
                target_function,
                net,
                rounds = 10,
                population_size = 10,
                lr = 0.005,
                lr_decay = 1,
                frame_duration = 0.1,
                num_samples = 300,
                x_range = [-PI, PI],
                smooth = True,
                show_loss = False):
        ax = Axes(
        x_range=x_range, y_range=[-3, 3],axis_config={"include_tip": False}
        ).scale(1.3)

        # Create the dataset
        x_data = np.random.uniform(x_range[0], x_range[1], num_samples)
        y_data = target_function(x_data)

        criteron = nn.MSELoss()

        def approx_func(x):
            return net(torch.tensor([[x]], dtype=torch.float32)).squeeze().cpu().detach().numpy()

        # Draw the target function
        target_graph = ax.plot(target_function, x_range=x_range, color=PURPLE, use_smoothing=smooth)
        scene.play(Create(target_graph))
        scene.wait()
        scene.play(FadeOut(target_graph))
        points = [ax.c2p(x, y) for x, y in zip(x_data, y_data)]
        data_points = [Dot(point=coord, color=RED, radius=0.05) for coord in points]
        data_points = VGroup(*data_points)
        scene.play(Create(data_points))

        # Draw the initial approximated function
        approx_graph = ax.plot(approx_func, x_range=x_range, color=BLUE, use_smoothing=smooth)
        scene.play(Create(approx_graph))

        inputs = torch.Tensor([[x] for x in x_data])
        true_outputs = torch.Tensor([[y] for y in y_data])
        best_loss = criteron(net(inputs), true_outputs)
        print(f'Initial Loss: {best_loss.item():.4f}')
        if show_loss:
            loss_label = MathTex('Loss: ', '{:.4f}'.format(best_loss.item())).to_corner(UL)
            scene.add(loss_label)

        for round in range(rounds):
            new_best = False
            best = net

            for child in range(population_size):
                net_child = copy.deepcopy(net)
                mutation = random.randint(0, 1)
                if mutation == 0:
                    # add a small random change to a subset of the params
                    params = list(net_child.parameters())
                    params = random.sample(params, random.randint(0, len(params)))
                    for param in params:
                        mutator = torch.randn_like(param.data) * lr
                        filter_prob = random.uniform(0, 1)
                        mutator = torch.where(torch.rand_like(mutator) < filter_prob, mutator, torch.zeros_like(mutator))
                        param.data = param.data + mutator
                else:
                    # add a small random change to a single random param
                    params = list(net_child.parameters())
                    param = random.choice(params)
                    idx = tuple(torch.randint(0, s, (1,)).item() for s in param.shape)
                    param.data[idx] += torch.randn(1).item() * lr

                net_child.eval()

                loss = criteron(net_child(inputs), true_outputs)
                if loss < best_loss:
                    best = net_child
                    best_loss = loss
                    new_best = True
            if new_best:
                # redraw the approximated function
                net = best
                new_approx_graph = ax.plot(approx_func, x_range=x_range, color=BLUE, use_smoothing=smooth)
                scene.play(ReplacementTransform(approx_graph, new_approx_graph), run_time=frame_duration, rate_func=linear)
                approx_graph = new_approx_graph
                scene.add(approx_graph)

                if show_loss:
                    scene.remove(loss_label)
                    loss_label = MathTex('Loss: ', '{:.4f}'.format(best_loss.item())).to_corner(UL)
                    scene.add(loss_label)

            print(f'Round: {round} | Current Loss: {best_loss.item():.4f} | Lr: {lr:.4f}')    
            lr *= lr_decay
        scene.wait()

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()
    
    def forward(self, x):
        x[x>=0] = 1
        x[x<0] = -1
        return x
    
class EvolveSimpleTest(Scene):
    def construct(self):
        # net = SimpleNN(hidden_size=10, hidden_layers=4, activation=BinaryActivation)
        # net = SkipConn(hidden_size=20, hidden_layers=4)
        # net = Fourier(fourier_order=4, hidden_size=5, hidden_layers=2)
        net = RecurrentNN(hidden_size=10, hidden_layers=4, iterations=4)
        # net = AdaptiveExpertNN(hidden_size=20, num_experts=2, fourier_order=1)
        # net = KANLinear(in_features=1, out_features=1, grid_size=5, spline_order=3, scale_noise=0.0, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])

        def sine(x):
            return np.sin(3*x)

        EvolveCurve(self, sine, net,
                    rounds=30,
                    population_size=1000,
                    lr=0.1,
                    lr_decay=0.98,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    smooth=False)


class EvolveBinary(Scene):
    def construct(self):
        net = SimpleNN(hidden_size=10, hidden_layers=4, activation=BinaryActivation)

        def sine(x):
            return np.sin(3*x)

        EvolveCurve(self, sine, net,
                    rounds=50,
                    population_size=1000,
                    lr=1,
                    lr_decay=0.98,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    smooth=False)
        

class EvolveMassive(Scene):
    def construct(self):
        net = SkipConn(hidden_size=100, hidden_layers=20, activation=nn.LeakyReLU)
        # print the number of parameters
        print("Number of parameters: ", sum(p.numel() for p in net.parameters()))
        def sine(x):
            return np.sin(6*x) + np.cos(4*x) + np.sin(2*x) + np.cos(6*x)

        EvolveCurve(self, sine, net,
                    rounds=300,
                    population_size=1000,
                    lr=1,
                    lr_decay=0.98,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    smooth=True)

