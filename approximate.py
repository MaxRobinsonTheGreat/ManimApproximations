import torch
from torch import nn, optim
import numpy as np
from manim import *
import random


# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, in_size=1, out_size=1, hidden_size=5, hidden_layers=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()) for _ in range(hidden_layers)],
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.layers(x)

class NetworkLearning(Scene):
    def construct(self):
        epochs = 50
        lr = 0.01
        frame_rate = 20
        num_samples = 200
        hidden_size = 5
        hidden_layers = 2
        x_min = -2 * np.pi
        x_max = 2 * np.pi

        # Define the target function
        def target_function(x):
            return np.sin(x)

        # Create the dataset
        x_data = np.linspace(x_min, x_max, num_samples)
        y_data = target_function(x_data)

        # Initialize the network, loss function and optimizer
        net = SimpleNN(hidden_size=hidden_size, hidden_layers=hidden_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # Transform the data for visualization
        x_graph = x_data # / (2 * np.pi) * 5
        y_graph_target = y_data 
        y_graph_approx = net(torch.Tensor(x_data).view(-1, 1)).detach().numpy().reshape(-1)

        # Draw the target function
        target_graph = self.get_graph(x_graph, y_graph_target, color=BLUE)
        self.play(Create(target_graph))

        # Draw the sampled data points
        data_points = self.get_data_points(x_graph, y_graph_target)
        self.play(FadeIn(data_points))

        # Draw the initial approximated function
        approx_graph = self.get_graph(x_graph, y_graph_approx, color=RED)
        self.play(Create(approx_graph))

        # Start the training process
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            for i in perm:
                inputs = torch.Tensor([[x_data[i]]])
                labels = torch.Tensor([[y_data[i]]])

                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % frame_rate == 0:  # Update the graph every 20 samples
                    y_graph_approx = net(torch.Tensor(x_data).view(-1, 1)).detach().numpy().reshape(-1)
                    new_approx_graph = self.get_graph(x_graph, y_graph_approx, color=RED)
                    self.play(Transform(approx_graph, new_approx_graph), run_time=0.1, rate_func=linear)

        self.wait()

    def get_graph(self, x_data, y_data, color=WHITE):
        points = [np.array([x, y, 0]) for x, y in zip(x_data, y_data)]
        line_graph = VMobject()
        line_graph.set_points_as_corners(points)
        line_graph.set_color(color)
        return line_graph

    def get_data_points(self, x_data, y_data):
        points = [Dot(point=[x, y, 0], color=YELLOW, radius=0.05) for x, y in zip(x_data, y_data)]
        return VGroup(*points)


class SphereSurface(ThreeDScene):
    def construct(self):
        # Define the parametric surface function for a sphere
        scalar = 2
        def sphere(u, v):
            return np.array([
                scalar * np.cos(u) * np.cos(v),
                scalar * np.cos(u) * np.sin(v),
                scalar * np.sin(u)
            ])

        # Create the parametric surface
        surface = Surface(
            sphere,
            resolution=(20, 20),
            u_range=[-PI / 2, PI / 2],
            v_range=[0, TAU],
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        # Display the surface
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.add(surface)
        self.wait(1)


class SphereApproximation(ThreeDScene):
    def construct(self):
        # Define the parametric surface function for a sphere
        size = 3
        epochs = 50
        lr = 0.001
        num_samples = 500
        num_display_samples = 100
        hidden_size = 20
        hidden_layers = 5
        step_size = 15

        def sphere(u, v):
            return np.array([
                size * np.cos(u) * np.cos(v),
                size * np.cos(u) * np.sin(v),
                size * np.sin(u)
            ])

        # Create the parametric surface that is slightly transparent
        true_surface = Surface(
            sphere,
            resolution=(20, 20),
            u_range=[-PI / 2, PI / 2],
            v_range=[0, TAU],
            checkerboard_colors=[PURPLE_D, PURPLE_E],
        ).set_opacity(0.9)

        # Display the surface
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=-0.1)
        self.play(Create(true_surface))
        self.wait(1)
        
        u_data = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1) - np.pi / 2
        v_data = np.random.uniform(0, 2 * np.pi, num_samples)
        # draw from a normal distribution but only keep the values within the range [-pi/2, pi/2] and [0, 2pi]
        # u_data = np.random.normal(0, np.pi / 2, num_samples)
        # v_data = np.random.normal(np.pi, np.pi, num_samples)  
        x_data, y_data, z_data = sphere(u_data, v_data)

        data_points = [np.array([x, y, z]) for x, y, z in zip(x_data, y_data, z_data)]
        display_points = random.sample(data_points, num_display_samples)
        print(len(display_points), len(data_points))

        # Draw the sampled data points
        self.play(*[FadeIn(Dot3D(point=d, color=RED, radius=0.05, resolution=[2,2])) for d in display_points])
        self.play(FadeOut(true_surface))
        # self.wait(5)

        net = SimpleNN(in_size=2, out_size=3, hidden_size=hidden_size, hidden_layers=hidden_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

        def shift_values(u, v):
            # shift u and v from [-pi/2, pi/2] and [0, 2pi] to [-1, 1]
            return [u / (PI / 2), (v / (TAU))-1]
            # return [u, v]

        def approx_sphere(u, v):
            inputs = shift_values(u, v)
            return net(torch.Tensor([inputs])).detach().numpy().reshape(-1)
        
        approx_surface = Surface(
            approx_sphere,
            resolution=(20, 20),
            u_range=[-PI / 2, PI / 2],
            v_range=[0, TAU],
            checkerboard_colors=[BLUE_D, BLUE_E],
        ).set_opacity(0.9)

        self.add(approx_surface)

        for epoch in range(epochs):
            random_samples = np.random.permutation(num_samples)
            tot_loss = 0
            for i in random_samples:
                inputs = torch.Tensor([shift_values(u_data[i], v_data[i])])
                labels = torch.Tensor([[x_data[i], y_data[i], z_data[i]]])

                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            print('Epoch: {}, Loss: {}'.format(epoch, tot_loss / num_samples))
            
            new_approx_surface = Surface(
                approx_sphere,
                resolution=(20, 20),
                u_range=[-PI / 2, PI / 2],
                v_range=[0, TAU],
                checkerboard_colors=[BLUE_D, BLUE_E],
            ).set_opacity(0.9)
            self.play(Transform(approx_surface, new_approx_surface), run_time=0.2, rate_func=linear)
            scheduler.step()

        self.wait(1)
