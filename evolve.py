import torch
from torch import nn
import numpy as np
from manim import *
import random
import copy
import math
from tqdm import tqdm
from models import SimpleNN, SkipConn, Fourier, SimpleTaylorNN, TaylorNN, RecurrentNN, AdaptiveExpertNN
from shared import create_weights_matricies
from kan import KANLinear

# Use local search to evolve the network
def EvolveCurve(scene,
                target_function,
                net,
                init_search_rounds = 10,
                rounds = 10,
                population_size = 10,
                lr = 0.005,
                lr_decay = 1,
                frame_rate = 5,
                frame_duration = 0.1,
                num_samples = 300,
                x_range = [-PI, PI],
                smooth = True,
                show_loss = False, 
                show_weights = False):
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
        approx_graph = ax.plot(approx_func, x_range=x_range, color=BLUE)
        scene.play(Create(approx_graph))

        if show_loss:
            loss_label = MathTex('Loss: ', '{:.4f}'.format(0)).scale(1.5).to_corner(UL)
            scene.add(loss_label)

        if show_weights:
            weights_matricies = create_weights_matricies(net)
            scene.add(weights_matricies)

        net = net
        inputs = torch.Tensor([[x] for x in x_data])
        true_outputs = torch.Tensor([[y] for y in y_data])
        best_loss = criteron(net(inputs), true_outputs)
        print(f'Initial Loss: {best_loss.item():.4f}')
        for _ in tqdm(range(init_search_rounds), desc='Random Search Initialization...'):
            random_net = copy.deepcopy(net)
            # Reset weights using default PyTorch initialization
            for param in random_net.parameters():

                if len(param.shape) > 1:
                    # For weight matrices, use uniform initialization
                    fan_in = param.shape[0] if len(param.shape) > 0 else 1
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(param, -bound, bound)
                else:
                    # For bias vectors, use uniform initialization
                    fan_in = param.shape[0] if len(param.shape) > 0 else 1
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(param, -bound, bound)

            loss = criteron(random_net(inputs), true_outputs)
            if loss < best_loss:
                best_loss = loss
                net = random_net

        print(f'Random Search Loss: {best_loss.item():.4f}')

        for round in range(rounds):
            new_best = False
            best = net

            for child in range(population_size):
                net_child = copy.deepcopy(net)
                # add a small random change to a subset of the params
                # get random subset of params
                params = list(net_child.parameters())
                params = random.sample(params, random.randint(0, len(params)))
                for param in params:
                    mutator = torch.randn_like(param.data) * lr
                    filter_prob = random.uniform(0, 1)
                    mutator = torch.where(torch.rand_like(mutator) < filter_prob, mutator, torch.zeros_like(mutator))
                    param.data = param.data + mutator
                net_child.eval()

                loss = criteron(net_child(inputs), true_outputs)
                if loss < best_loss:
                    best = net_child
                    best_loss = loss
                    new_best = True
            if new_best:
                # redraw the approximated function
                net = best
                new_approx_graph = ax.plot(approx_func, x_range=x_range, color=BLUE)
                scene.play(ReplacementTransform(approx_graph, new_approx_graph), run_time=frame_duration, rate_func=linear)
                approx_graph = new_approx_graph
                scene.add(approx_graph)
            print(f'Round: {round} | Current Loss: {best_loss.item():.4f} | Lr: {lr:.4f}')    
            lr *= lr_decay
        scene.wait()


class EvolveSimpleTest(Scene):
    def construct(self):
        net = SimpleNN(hidden_size=10, hidden_layers=4)
        # net = SkipConn(hidden_size=20, hidden_layers=4)
        # net = Fourier(fourier_order=4, hidden_size=5, hidden_layers=2)
        # net = RecurrentNN(hidden_size=10, hidden_layers=4, iterations=4)
        # net = AdaptiveExpertNN(hidden_size=20, num_experts=2, fourier_order=1)
        # net = KANLinear(in_features=1, out_features=1, grid_size=5, spline_order=3, scale_noise=0.0, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])

        def sine(x):
            return np.sin(3*x)

        EvolveCurve(self, sine, net,
                    init_search_rounds=100,
                    rounds=30,
                    population_size=1000,
                    lr=0.1,
                    lr_decay=0.98,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    smooth=False)