import torch
from torch import nn, optim
import numpy as np
from manim import *
import random
from models import SimpleNN, SkipConn, Fourier, SimpleTaylorNN, TaylorNN, RecurrentNN, AdaptiveExpertNN
from shared import create_weights_matricies
from kan import KANLinear

def LearnCurve( scene,
                target_function,
                net,
                epochs = 10,
                lr = 0.005,
                batch_size = 20,

                frame_rate = 5,
                frame_duration = 0.1,
                num_samples = 300,
                x_range = [-PI, PI],
                sched_step = 10,
                smooth = True,
                show_loss = False, 
                show_weights = False,):

        ax = Axes(
            x_range=x_range, y_range=[-3, 3],axis_config={"include_tip": False}
        ).scale(1.3)

        # Create the dataset
        x_data = np.random.uniform(x_range[0], x_range[1], num_samples)
        y_data = target_function(x_data)

        optimizer = optim.Adam(net.parameters(), lr=lr)
        # criteron = lambda x, y: torch.mean(torch.abs(x - y))
        criteron = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=0.5)

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

        # Start the training process
        for epoch in range(epochs):
            batches = np.array_split(np.random.permutation(num_samples), batch_size)
            running_loss = 0
            for i, indices in enumerate(batches):
                inputs = torch.Tensor([[x] for x in x_data[indices]])
                labels = torch.Tensor([[y] for y in y_data[indices]])

                # Forward pass
                outputs = net(inputs)
                # compute average absolute difference between outputs and labels
                loss = criteron(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % frame_rate == 0: 
                    new_approx_graph = ax.plot(approx_func, x_range=x_range, color=BLUE)
                    scene.play(ReplacementTransform(approx_graph, new_approx_graph), run_time=frame_duration, rate_func=linear)
                    approx_graph = new_approx_graph
                    scene.add(approx_graph)

                    if show_loss:
                        scene.remove(loss_label)
                        avg_loss = running_loss / (i+1)
                        loss_label = MathTex('Loss: ', '{:.4f}'.format(avg_loss)).scale(1.5).to_corner(UL)
                        scene.add(loss_label)

                    if show_weights:
                        scene.remove(weights_matricies)
                        weights_matricies = create_weights_matricies(net)
                        scene.add(weights_matricies)

            print('Epoch: {}, Loss: {:.10f}'.format(epoch, running_loss / num_samples))
            scheduler.step()
        scene.wait()
        scene.remove(ax, approx_graph, data_points)


class LearnSimpleTest(Scene):
    def construct(self):
        net = SimpleNN(hidden_size=20, hidden_layers=4, activation=nn.GELU)
        # net = SkipConn(hidden_size=10, hidden_layers=4)
        # net = SkipConn(hidden_size=50, hidden_layers=7)
        # net = AdaptiveExpertNN(hidden_size=20, num_experts=2, fourier_order=1)
        # net = KolmogorovNetwork(in_size=1, out_size=1, inner_size=64, num_inner_funcs=5)
        # net = KANLinear(in_features=1, out_features=1, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])

        def sine(x):
            return np.sin(3*x)

        LearnCurve(self, sine, net,
                    epochs=10,
                    lr=0.01,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=10,
                    smooth=False)

class LearnSimple(Scene):
    def construct(self):
        net = SkipConn(hidden_size=50, hidden_layers=7)

        def sine(x):
            return np.sin(3*x)

        LearnCurve(self, sine, net,
                    epochs=30,
                    lr=0.001,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=10,
                    smooth=True)
        
        def wavey(x):
            return np.sin(2*x) - np.cos(3*x)
        
        LearnCurve(self, wavey, net,
            epochs=10,
            lr=0.001,
            batch_size=20,
            frame_rate=5,
            frame_duration=0.2,
            num_samples=300,
            x_range=[-PI, PI],
            sched_step=10,
            smooth=True)

        def gassian(x):
            return 2*np.exp(-x**2)
        
        LearnCurve(self, gassian, net,
                    epochs=10,
                    lr=0.001,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=5,
                    smooth=True)
        
        def cubic(x):
            return x**3/5
        
        LearnCurve(self, cubic, net,
            epochs=10,
            lr=0.001,
            batch_size=20,
            frame_rate=5,
            frame_duration=0.2,
            num_samples=300,
            x_range=[-PI, PI],
            sched_step=5,
            smooth=True)
        
        def piecewise(x):
            return -abs(x)/4 + abs(2+x) - abs(2*x+1)/2
        
        LearnCurve(self, piecewise, net,
                epochs=10,
                lr=0.001,
                batch_size=20,
                frame_rate=5,
                frame_duration=0.2,
                num_samples=300,
                x_range=[-PI, PI],
                sched_step=5,
                smooth=False)
        

class LearnTiny(Scene):
    def construct(self):
        net = SimpleNN(hidden_size=2, hidden_layers=0)

        def target_function(x):
            return abs(x)

        LearnCurve(self, target_function, net,
                    epochs=10,
                    lr=0.01,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=200,
                    x_range=[-PI, PI],
                    sched_step=10,
                    smooth=False,
                    show_weights=True)


class LearnRecurrent(Scene):
    def construct(self):
        net = RecurrentNN(hidden_size=50, hidden_layers=5, iterations=2)

        def target_function(x):
            return np.sin(2*x) - np.cos(3*x)

        LearnCurve(self, target_function, net,
                    epochs=50,
                    lr=0.001,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=200,
                    x_range=[-PI, PI],
                    sched_step=10)


class LearnWithLoss(Scene):
    def construct(self):
        net = SkipConn(hidden_size=100, hidden_layers=10)

        def target(x):
            return 2*abs(np.cos(2*x))-1

        LearnCurve(self, target, net,
                    epochs=30,
                    lr=0.001,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=10,
                    smooth=True,
                    show_loss=True)
        
        
class LearnTaylor(Scene):
    def construct(self):
        net = SimpleTaylorNN(taylor_order=8)

        def target_function(x):
            return np.sin(2*x) - np.cos(3*x)

        LearnCurve(self, target_function, net,
                    epochs=50,
                    lr=0.01,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=10,
                    smooth=True)

        net = TaylorNN(taylor_order=8, hidden_size=100, hidden_layers=7)
        LearnCurve(self, target_function, net,
                    epochs=50,
                    lr=0.01,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=10,
                    smooth=True)
        self.wait()


class LearnFourier(Scene):
    def construct(self):
        net = Fourier(fourier_order=16, hidden_size=100, hidden_layers=7)

        def target_function(x):
            return abs(x)/4 - abs(2+x) + abs(3*x+3)/2 - 1

        LearnCurve(self, target_function, net,
                    epochs=20,
                    lr=0.001,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=300,
                    x_range=[-PI, PI],
                    sched_step=10,
                    smooth=False)


class FourierOverfit(Scene):
    def construct(self):
        net = Fourier(fourier_order=32, hidden_size=50, hidden_layers=5)

        def target_function(x):
            return abs(x)/4 - abs(2+x) + abs(3*x+3)/2 - 1

        LearnCurve(self, target_function, net,
                    epochs=50,
                    lr=0.001,
                    batch_size=20,
                    frame_rate=5,
                    frame_duration=0.2,
                    num_samples=100,
                    x_range=[-PI, PI],
                    sched_step=15,
                    smooth=False)


class SphereSurface(ThreeDScene):
    def construct(self):
        # Define the parametric surface function for a sphere
        scalar = 3
        def sphere(u, v):
            return np.array([
                scalar * np.cos(u) * np.cos(v),
                scalar * np.cos(u) * np.sin(v),
                scalar * np.sin(u)
            ])

        # Create the parametric surface
        surface = Surface(
            sphere,
            resolution=(40, 40),
            u_range=[-PI / 2, PI / 2],
            v_range=[0, TAU],
            checkerboard_colors=[PURPLE_D, PURPLE_E],
        ).set_opacity(0.9)

        # Display the surface
        self.set_camera_orientation(phi=80 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=-0.1)
        self.play(Create(surface), run_time=2)
        self.wait(10)


def shift_value(x, start_range, end_range):
    return (x - start_range[0]) / (start_range[1] - start_range[0]) * (end_range[1] - end_range[0]) + end_range[0]

def generate_inputs(u_data, v_data, u_range, v_range, nn_range=None):
    if nn_range is not None:
        u_data = shift_value(u_data, u_range, nn_range)
        v_data = shift_value(v_data, v_range, nn_range)
    inputs = torch.Tensor(np.array([u_data, v_data]).T)
    return inputs

def generate_outputs(x, y, z):
    return torch.Tensor(np.array([x, y, z]).T)

def generate_display_points(x_data, y_data, z_data, num_display_samples):
    data_points = [np.array([x, y, z]) for x, y, z in zip(x_data, y_data, z_data)]
    return random.sample(data_points, num_display_samples)

def precomute_net_outputs(scene, net, u_range, v_range, resolution):
    # precomputes the outputs of the network for each point in the resolution for a Manim Surface
    # returns a function that takes in u, v and returns the output of the network

    # first create an empty list of inputs and make a dummy function that saves the inputs to the list and returns 0,0,0
    # then add the dummy surface to the scene and remove it which will save the inputs to the list
    # then use the list of inputs to generate the outputs of the network
    # then create a new function that takes in u, v and returns the precomputed output of the network
    # return the function

    # this is hacky but dramatically speeds up the animation as we can now batch the network inputs
    inputs = []
    def dummy_func(u, v):
        inputs.append([u, v])
        return np.array([0., 0., 0.])
    
    dummy_surface = Surface(
        dummy_func,
        resolution=resolution,
        u_range=u_range,
        v_range=v_range,
        checkerboard_colors=[BLUE_D, BLUE_E],
    ).set_opacity(0)

    scene.add(dummy_surface)
    scene.remove(dummy_surface)

    net_inputs = torch.Tensor(inputs)
    outputs = net(net_inputs).detach().numpy()

    # print the largest and smallest inputs
    # print('largest input: ', inputs.max(dim=0))
    # print('smallest input: ', inputs.min(dim=0))
    
    input_output_map = {}
    for i in range(len(inputs)):
        input_output_map[(inputs[i][0].item(), inputs[i][1].item())] = outputs[i]

    def approx_surface_func(u, v):
        return input_output_map[(u, v)]
    
    return approx_surface_func



def approximate_surface(scene,
                       net,
                       inputs,
                       outputs,
                       nn_range,
                       resolution,
                       epochs = 10,
                       lr = 0.001,
                       batch_size = 20,
                       sched_step_size = None,
                       ):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=0.5)

    num_samples = len(inputs)
    
    approx_surface_func = precomute_net_outputs(scene, net, nn_range, nn_range, resolution)
    
    approx_surface = Surface(
        approx_surface_func,
        resolution=resolution,
        u_range=nn_range,
        v_range=nn_range,
        checkerboard_colors=[BLUE_D, BLUE_E],
    ).set_opacity(0.9)

    scene.add(approx_surface)

    for epoch in range(epochs):
        index_batches = np.array_split(np.random.permutation(num_samples), batch_size)
        tot_loss = 0
        for i in index_batches:
            ins = inputs[i]
            outs = outputs[i]

            # Forward pass
            pred = net(ins)
            loss = criterion(pred, outs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Epoch: {}, Loss: {:.10f}'.format(epoch, tot_loss / num_samples))
        scheduler.step()
        approx_surface_func = precomute_net_outputs(scene, net, nn_range, nn_range, resolution)
        
        new_approx_surface = Surface(
            approx_surface_func,
            resolution=resolution,
            u_range=nn_range,
            v_range=nn_range,
            checkerboard_colors=[BLUE_D, BLUE_E],
        ).set_opacity(0.9)
        scene.play(ReplacementTransform(approx_surface, new_approx_surface), run_time=0.3, rate_func=linear)
        approx_surface = new_approx_surface
        scene.add(approx_surface)

    return approx_surface
    

class SphereApproximation(ThreeDScene):
    def construct(self):
        # Define the parametric surface function for a sphere
        epochs = 15
        lr = 0.001
        batch_size = 20
        num_samples = 1000
        num_display_samples = 200
        hidden_size = 50
        hidden_layers = 7
        nn_range = [-PI, PI]
        step_size = 15

        size = 3
        resolution = (40, 40)
        u_range = [-PI / 2, PI / 2]
        v_range = [0, TAU]
        def sphere(u, v):
            return np.array([
                size * np.cos(u) * np.cos(v),
                size * np.cos(u) * np.sin(v),
                size * np.sin(u)
            ])

        # Create the parametric surface that is slightly transparent
        true_surface = Surface(
            sphere,
            resolution=resolution,
            u_range=u_range,
            v_range=v_range,
            checkerboard_colors=[PURPLE_D, PURPLE_E],
        ).set_opacity(0.9)
        # Display the surface
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=-0.1)

        
        u_data = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1) - np.pi / 2
        v_data = np.random.uniform(0, 2 * np.pi, num_samples)
        x_data, y_data, z_data = sphere(u_data, v_data)

        display_points = generate_display_points(x_data, y_data, z_data, num_display_samples)
        inputs = generate_inputs(u_data, v_data, u_range, v_range, nn_range)
        outputs = generate_outputs(x_data, y_data, z_data)

        # Draw the sampled data points
        # self.play(Create(true_surface))
        # self.wait(2)
        # dots = [Dot3D(point=d, color=RED, radius=0.05, resolution=[5,5]) for d in display_points]
        # self.play(*[FadeIn(d) for d in dots])
        # self.play(FadeOut(true_surface))
        # self.wait(1)

        # net = Fourier(in_size=2, out_size=3, fourier_order=8, hidden_size=hidden_size, hidden_layers=hidden_layers)
        net = SkipConn(in_size=2, out_size=3, hidden_size=hidden_size, hidden_layers=hidden_layers, activation=nn.GELU)
        # net = SimpleNN(in_size=2, out_size=3, hidden_size=hidden_size, hidden_layers=hidden_layers, activation=nn.GELU)
        # net = KANLinear(in_features=2, out_features=3, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
        
        print('before')
        approx_surface = approximate_surface(self,
                                            net,
                                            inputs,
                                            outputs,
                                            nn_range,
                                            resolution=resolution,
                                            epochs=epochs,
                                            lr=lr,
                                            batch_size=batch_size,
                                            sched_step_size=step_size)


def get_spiral_shell():
    r = 1
    a = 1.25
    b = 1.25
    c = 1
    d = 3.5
    e = 0
    f = 0.17
    h = -1
    def spiral_shell(u, v):
        exp = pow(np.e, f*u)

        x = r*exp * (-1.4*e + b*np.sin(v))
        y = r*exp * (d + a*np.cos(v)) * np.sin(c*u)
        z = r*exp * (d + a*np.cos(v)) * np.cos(c*u) + h

        return np.array([x, y, z])
    return spiral_shell


class SpiralSurface(ThreeDScene):
    def construct(self):
        # create a parametric surface using the spiral shell function for 10 seconds resolution 60, 60
        u_range = [-25, 0]
        v_range = [-2*PI, 2*PI]
        surface = Surface(
            get_spiral_shell(),
            resolution=(60, 60),
            u_range=u_range,
            v_range=v_range,
            checkerboard_colors=[GOLD, GOLD_E],
            stroke_color=GOLD_E
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)

        self.play(Create(surface), run_time=2)
        self.wait(15)


class SpiralApproximation(ThreeDScene):
    def construct(self):
        epochs = 20
        lr = 0.001
        batch_size = 20
        num_samples = 3000
        num_display_samples = 150
        hidden_size = 300
        hidden_layers = 30
        step_size = 15
        nn_range = [-1, 1]

        u_range = [-25, 0]
        v_range = [-2*PI, 2*PI]
        resolution=(40, 40)

        self.set_camera_orientation(phi=75 * DEGREES, theta=50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)
    
        # spiral shell parameters
        spiral_shell = get_spiral_shell()

        # Create the parametric surface
        surface = Surface(
            spiral_shell,
            resolution=resolution,
            u_range=u_range,
            v_range=v_range,
            checkerboard_colors=[GOLD, GOLD_E],
            stroke_color=GOLD_E
        )

        u_data = -np.random.exponential(scale=7, size=num_samples)
        u_data = np.clip(u_data, u_range[0], u_range[1])
        v_data = np.random.uniform(v_range[0], v_range[1], num_samples)
        x_data, y_data, z_data = spiral_shell(u_data, v_data)
        
        display_points = generate_display_points(x_data, y_data, z_data, num_display_samples)
        inputs = generate_inputs(u_data, v_data, u_range, v_range, nn_range)
        outputs = generate_outputs(x_data, y_data, z_data)
        
        self.play(Create(surface))
        self.wait(2)
        dots = [Dot3D(point=d, color=RED, radius=0.05, resolution=[5,5]) for d in display_points]
        self.play(*[FadeIn(d) for d in dots])
        self.wait()
        self.play(FadeOut(surface), run_time=1)

        # net = SkipConn(in_size=2, out_size=3, hidden_size=hidden_size, hidden_layers=hidden_layers)
        net = Fourier(in_size=2, out_size=3, fourier_order=16, hidden_size=hidden_size, hidden_layers=hidden_layers)

        approx_surface = approximate_surface(self, 
                                            net, 
                                            inputs, 
                                            outputs, 
                                            nn_range,
                                            resolution=resolution,
                                            epochs=epochs, 
                                            lr=lr, 
                                            batch_size=batch_size, 
                                            sched_step_size=step_size)
        self.wait(10)


class CubeApproximation(ThreeDScene):
    def construct(self):
        epochs = 50
        lr = 0.001
        batch_size = 20
        num_samples = 500
        num_display_samples = 50
        hidden_size = 100
        hidden_layers = 10
        step_size = 10
        nn_range = [-1, 1]

        u_range = [-1, 1]
        v_range = [-1, 1]
        resolution=(20, 20)

        self.set_camera_orientation(phi=75 * DEGREES, theta=50 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)

        range_size = u_range[1]-u_range[0]
        num_faces = 4
        step_size = range_size / num_faces
        offset = range_size/2
        # def cube(u, v):
        #     if (u <= -step_size):
        #         return np.array([(u+step_size)*num_faces, v, offset])
        #     if (u <= 0):
        #         return np.array([0, v, -(u+step_size)*num_faces + offset])
        #     if (u <= step_size):
        #         return np.array([-u*num_faces, v, -num_faces/2 + offset])
        #     return np.array([-num_faces/2, v, (u-step_size*2)*num_faces + offset])
        
        def cube(u, v):
            u_scalar = np.isscalar(u)
            v_scalar = np.isscalar(v)
            
            u = np.array([u]) if u_scalar else u
            v = np.array([v]) if v_scalar else v

            x1 = np.where(u <= -step_size, (u + step_size) * num_faces, 0)
            x2 = np.where(np.logical_and(-step_size < u, u <= 0), 0, x1)
            x3 = np.where(np.logical_and(0 < u, u <= step_size), -u * num_faces, x2)
            x = np.where(u > step_size, -num_faces / 2, x3)

            z1 = np.where(u <= -step_size, offset, 0)
            z2 = np.where(np.logical_and(-step_size < u, u <= 0), -(u + step_size) * num_faces + offset, z1)
            z3 = np.where(np.logical_and(0 < u, u <= step_size), -num_faces / 2 + offset, z2)
            z = np.where(u > step_size, (u - step_size * 2) * num_faces + offset, z3)

            result = np.array([x, v, z])

            if u_scalar and v_scalar:
                result = result.flatten()

            return result


        
        # Create the parametric surface
        surface = Surface(
            cube,
            resolution=resolution,
            u_range=u_range,
            v_range=v_range,
            checkerboard_colors=[BLUE_D, BLUE_E],
            should_make_jagged=True
        )

        u_data = np.random.uniform(u_range[0], u_range[1], num_samples)
        v_data = np.random.uniform(v_range[0], v_range[1], num_samples)
        x_data, y_data, z_data = cube(u_data, v_data)
        inputs = generate_inputs(u_data, v_data, u_range, v_range, nn_range)
        outputs = generate_outputs(x_data, y_data, z_data)

        display_points = generate_display_points(x_data, y_data, z_data, num_display_samples)

        self.play(Create(surface))
        self.wait(2)
        self.add(*[Dot3D(point=d, color=RED, radius=0.05, resolution=[2,2]) for d in display_points])
        self.wait()
        self.play(Uncreate(surface), run_time=2)

        net = SkipConn(in_size=2, out_size=3, hidden_size=hidden_size, hidden_layers=hidden_layers)

        approx_surface = approximate_surface(self,
                                            net,
                                            inputs,
                                            outputs,
                                            nn_range,
                                            resolution=resolution,
                                            epochs=epochs,
                                            lr=lr,
                                            batch_size=batch_size,
                                            sched_step_size=step_size)
        self.wait(1)

