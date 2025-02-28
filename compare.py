import torch
from torch import nn, optim
import numpy as np
from manim import *
from models import SimpleNN, SkipConn
from kan import KANLinear

from approximate import LearnCurve
from evolve import EvolveCurve

net = SimpleNN(hidden_size=20, hidden_layers=4, activation=nn.LeakyReLU)
# net = SkipConn(hidden_size=20, hidden_layers=4, activation=nn.LeakyReLU)
# print number of parameters
print("Number of parameters: ", sum(p.numel() for p in net.parameters()))

def target(x):
    return np.sin(3*x)

class Evolve(Scene):
    def construct(self):
        EvolveCurve(self, target, net,
                    rounds=150,
                    population_size=1000,
                    lr=1,
                    lr_decay=0.98,
                    frame_duration=0.3,
                    num_samples=300,
                    x_range=[-PI, PI],
                    show_loss=True,
                    smooth=True)

class SGD(Scene):
    def construct(self):
        LearnCurve(self, target, net,
                    epochs=100,
                    lr=0.02,
                    batch_size=20,
                    optimizer="sgd",
                    frame_rate=20,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=10,
                    sched_gamma = 0.8,
                    show_loss=True, 
                    smooth=True)

class Adam(Scene):
    def construct(self):
        LearnCurve(self, target, net,
                    epochs=100,
                    lr=0.01,
                    optimizer="adam",
                    frame_rate=20,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=20,
                    show_loss=True,
                    smooth=True)


