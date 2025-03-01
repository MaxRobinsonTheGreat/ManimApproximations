from manim import *
import numpy as np
import math

class MinMax(Scene):
    def construct(self):
        # Create axes (invisible but needed for coordinates)
        axes = Axes(
            x_range=[-PI, PI, 1],
            y_range=[-2, 4, 1],
            axis_config={"color": BLUE},
        ).scale(1.18).shift(UP)
        
        def func(x):
            return np.sin(2*x) - np.cos(x)

        # Create x-value display
        x_value = DecimalNumber(
            -PI,
            num_decimal_places=2,
            include_sign=True,
            font_size=36
        ).to_corner(UR)
        x_label = Text("x = ", font_size=36).next_to(x_value, LEFT)
        
        # Create the graph
        graph = axes.plot(func, color=BLUE)

        # Create a dot that will move along the curve
        dot = Dot(color=RED)
        
        # Starting position (x=0)
        start_point = axes.c2p(-PI, func(-PI))
        dot.move_to(start_point)

        # Add graph, dot, and x-value display to scene
        self.play(Create(graph))
        self.play(Create(dot), Write(x_label), Write(x_value))
        self.wait()

        # Function to update x_value during animations
        def update_x_value(mob):
            x = axes.p2c(dot.get_center())[0]  # Get x coordinate from dot position
            mob.set_value(x)

        x_value.add_updater(update_x_value)

        # Animate the dot's movement along the curve
        def get_points(start_x, end_x, num_points=100):
            return [axes.c2p(x, func(x)) 
                   for x in np.linspace(start_x, end_x, num_points)]
        
        # Move to right point
        self.play(MoveAlongPath(dot, 
            VMobject().set_points_as_corners(get_points(-PI, PI))), 
            run_time=4, rate_func=linear)
        self.wait()
        
        # Move to local maximum
        self.play(MoveAlongPath(dot, 
            VMobject().set_points_as_corners(get_points(PI, -2.506))), 
            run_time=2)
        self.wait()
        
        # Move to global minimum
        self.play(MoveAlongPath(dot, 
            VMobject().set_points_as_corners(get_points(-2.506, 2.138))), 
            run_time=2)
        self.wait()
        
        # Move to local minimum
        self.play(MoveAlongPath(dot, 
            VMobject().set_points_as_corners(get_points(2.138, -0.634))), 
            run_time=2)
        self.wait()


class ReluBinary(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 4, 1],
            axis_config={"color": BLUE},
        ).scale(1.19)

        def relu(x):
            return np.maximum(0, x)
        title = Text("ReLU").next_to(axes, UP).scale(0.9).shift(DOWN)
        graph = axes.plot(relu, color=BLUE, use_smoothing=False)
        func = MathTex(r"\text{max}(0, x)").next_to(axes, DOWN).shift(UP).scale(0.9)

        self.play(Create(graph), Write(func), Write(title))
        self.wait() 

        self.remove(graph, func, title)

        def binary(x):
            return 1 if x >= 0 else 0
        title = Text("Binary Step").next_to(axes, UP).shift(DOWN).scale(0.9)
        graph = axes.plot(binary, color=BLUE, use_smoothing=False)
        func = MathTex(r"\text{sign}(x)").next_to(axes, DOWN).shift(UP).scale(0.9)

        self.play(Create(graph), Write(func), Write(title))
        self.wait()

class ContinuousDifferentiable(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 4, 1],
            axis_config={"color": BLUE},
        ).scale(1.19)

        # Linear function
        def linear(x):
            return x + 1
        
        graph = axes.plot(lambda x: linear(x), color=BLUE)
        
        self.play(Create(graph))
        self.wait()

        # Add hole at x=0
        hole_point = axes.c2p(0, linear(0))
        hole = Circle(radius=0.2, color=BLUE, fill_color=BLACK, fill_opacity=1).move_to(hole_point)
        
        self.play(Create(hole), run_time=0.5)
        self.wait(0.5)
        # Break function at x=0 (create two separate pieces)
        left_graph = axes.plot(lambda x: linear(x), x_range=[-3, 0], color=BLUE)
        right_graph = axes.plot(lambda x: linear(x), x_range=[0, 3], color=BLUE)
        
        self.remove(hole)
        self.remove(graph)
        graph = VGroup(left_graph, right_graph)

        # Increase gap between pieces
        left_graph2 = axes.plot(lambda x: linear(x) + 1, x_range=[-3, 0], color=BLUE)
        right_graph2 = axes.plot(lambda x: linear(x) - 1, x_range=[0, 3], color=BLUE)
        
        self.play(
            ReplacementTransform(left_graph, left_graph2),
            ReplacementTransform(right_graph, right_graph2),
        )

        self.play(
            left_graph2.animate.shift(LEFT*2),
            right_graph2.animate.shift(RIGHT*2),
        )
        self.wait(0.2)

        left_graph3 = axes.plot(lambda x: -x, x_range=[-3, 0], color=BLUE)
        right_graph3 = axes.plot(lambda x: x, x_range=[0, 3], color=BLUE)

        self.play(
            ReplacementTransform(left_graph2, left_graph3),
            ReplacementTransform(right_graph2, right_graph3),
        )
        self.wait()


class SaddlePoint(ThreeDScene):
    def construct(self):
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=(-2, 2),
            y_range=(-2, 2),
            z_range=(-2, 2),
            x_length=6,
            y_length=6,
            z_length=6,
        ).scale(0.8)

        # Create saddle surface (z = x² - y²)
        surface = Surface(
            lambda u, v: np.array([u, v, (u**2 - v**2)/2]),
            u_range=(-3, 3),
            v_range=(-3, 3),
            resolution=(20, 20),
            checkerboard_colors=[BLUE_D, BLUE_E],
        ).scale(0.8)
        self.set_camera_orientation(phi=90*DEGREES, theta=90*DEGREES, focal_distance=100)
        
        # Add everything to the scene
        self.add(surface)
        
        # Initial view showing the minimum along one axis
        self.wait()
        
        def eas_in_sine(t):
            return 1 - np.cos(t * np.pi / 2)

        # Rotate camera to reveal the saddle point nature
        self.move_camera(
            phi=60 * DEGREES,
            theta=50 * DEGREES,
            run_time=3,
            rate_func=eas_in_sine 
        )
        self.begin_ambient_camera_rotation(rate=-0.2)
        self.wait(30)
        

# shared for all scenes
def nn(x, a, b):
    return np.tanh(a*x + b)

def target(x):
    return np.sin(x)

d1 = (-1.5, target(-1.5))
d2 = (0, target(0))
d3 = (1.5, target(1.5))

def loss_landscape(a, b):
    loss = 0
    for x, y in [d1, d2, d3]:
        pred = nn(x, a, b)
        loss += (pred - y)**2
    return loss / 3

def dL_da(a, b):
    loss = 0
    for x, y in [d1, d2, d3]:
        pred = nn(x, a, b)
        loss += 2*(pred - y)*x*(1/np.cosh(a*x + b))**2
    return loss / 3

def dL_db(a, b):
    loss = 0
    for x, y in [d1, d2, d3]:
        pred = nn(x, a, b)
        loss += 2*(pred - y)*(1/np.cosh(a*x + b))**2
    return loss / 3


class LossLandscapeFunction(Scene):
    def construct(self):
        a = 1.1
        b = 0.8

        # Create the NN function display
        nn_eq = MathTex(
            "a: ", str(a), ",\\,\\, b: ", str(b),
            ",\\,\\, \\text{NN}(x) = \\tanh(", "a", "x", "+", "b", ")"
        ).scale(1.3)  # Make the top text bigger
        nn_eq[1].set_color(GREEN)  # a value
        nn_eq[3].set_color(GREEN)  # b value
        nn_eq[5].set_color(GREEN)  # a in equation
        nn_eq[8].set_color(GREEN)  # b in equation
        nn_eq.to_edge(UP)

        # Create the data points display
        data_points = MathTex(
            "\\text{Data: }", 
            "(", f"{d1[0]}", ",", f"{d1[1]:.1f}", ")", 
            ",\\,\\, (", f"{d2[0]}", ",", f"{d2[1]:.1f}", ")", 
            ",\\,\\, (", f"{d3[0]}", ",", f"{d3[1]:.1f}", ")"
        )
        # Color the data points blue
        # Only color the numbers blue
        for i in [2, 4, 7, 9, 12, 14]:  
            data_points[i].set_color(BLUE)
        data_points.next_to(nn_eq, DOWN, buff=0.5).scale(1.3)

        # Add to scene
        self.play(Write(nn_eq))
        self.wait()
        self.play(Write(data_points))
        self.wait()

        # Create calculations for each data point
        calcs = VGroup()
        for i, (x, _) in enumerate([d1, d2, d3]):
            pred = np.tanh(a*x + b)
            calc = MathTex(
                "\\text{NN}(",
                f"{x}",
                ")",
                "=\\tanh(",
                "a",
                f"\\cdot({x})",
                "+",
                "b",
                ")",
                "\\rightarrow",
                f"{pred:.3f}"
            )
            calc[1].set_color(BLUE)   # first x value
            calc[4].set_color(GREEN)  # a
            calc[7].set_color(GREEN)  # b
            calc[5].set_color(BLUE)   # input x in calculation
            calc[10].set_color(GREEN)  # output
            calc.next_to(data_points, DOWN, buff=1 + i*0.7)
            calc.align_to(data_points, LEFT)  # Align to the left
            calcs.add(calc)
            calc[10].set_x(3.5)  # Set absolute x position while keeping y position
        
        # Display calculations one at a time
        for calc in calcs:
            self.play(Write(calc))
        self.wait()

        # Add loss function definition
        loss_def = MathTex(
            "\\text{MSE\\_Loss}(", 
            "\\text{predicted}", 
            ",", 
            "\\text{true}", 
            ") = \\text{avg}((",
            "\\text{pred}",
            "-",
            "\\text{true}",
            ")^2)"
        ).next_to(data_points, DOWN, buff=4.0).scale(1.2)
        
        # Color the predicted/pred terms green and true terms blue
        loss_def[1].set_color(GREEN)  # predicted
        loss_def[3].set_color(BLUE)   # true
        loss_def[5].set_color(GREEN)  # pred
        loss_def[7].set_color(BLUE)   # true

        self.play(Write(loss_def))
        self.wait()

        # Remove the loss function definition
        self.play(FadeOut(loss_def))
        
        anims = []
        # After displaying all calculations
        for i, (x, y) in enumerate([d1, d2, d3]):
            pred = np.tanh(a*x + b)
            # Transform just the final output into MSE calculation
            mse_output = MathTex(
                "\\text{MSE}(", f"{pred:.2f}", ",", f"{y:.1f}", ")"
            )
            mse_output[1].set_color(GREEN)  # pred value
            mse_output[3].set_color(BLUE)   # true value
            mse_output.move_to(calcs[i][10])  # Move to same position as original output
            mse_output.set_x(4.5)  # Move further right
            
            anims.append(Transform(calcs[i][10], mse_output))

        self.play(*anims)
        self.wait()
        
        # Calculate average MSE
        mse_values = []
        for x, y in [d1, d2, d3]:
            pred = np.tanh(a*x + b)
            mse = (pred - y)**2
            mse_values.append(mse)
        avg_mse = sum(mse_values) / len(mse_values)
        
        # Display average loss
        avg_text = MathTex(
            "\\text{Average Loss} = ",
            f"{avg_mse:.3f}"
        ).scale(1.5)
        avg_text.next_to(calcs, DOWN, buff=1.0)
        avg_text.align_to(data_points, RIGHT).shift(RIGHT*0.5)
        
        self.play(Write(avg_text))
        self.wait()
        self.play(Indicate(avg_text[1]))
        self.wait()

        # Fade out previous calculations
        self.play(
            *[FadeOut(mob) for mob in [*calcs, avg_text]]
        )
        
        # Create each line separately
        line1 = MathTex("\\text{LossLandscape}(", "a", ",", "b", ")", "=")
        line2 = MathTex("\\frac{1}{3}", "\\Big(", "\\big(\\tanh(", "a", "\\cdot(", f"{d1[0]}", ")+", "b", ")-(", f"{d1[1]:.3f}", ")\\big)^2", "+")
        line3 = MathTex("\\quad\\big(\\tanh(", "a", "\\cdot(", f"{d2[0]}", ")+", "b", ")-(", f"{d2[1]:.3f}", ")\\big)^2", "+")
        line4 = MathTex("\\quad\\big(\\tanh(", "a", "\\cdot(", f"{d3[0]}", ")+", "b", ")-(", f"{d3[1]:.3f}", ")\\big)^2", "\\Big)")
        
        # Color parameters green in each line
        line1[1].set_color(GREEN)
        line1[3].set_color(GREEN)
        line2[3].set_color(GREEN)  # a
        line2[7].set_color(GREEN)  # b
        for line in [line3, line4]:
            line[1].set_color(GREEN)  # a
            line[5].set_color(GREEN)  # b
            
        # Color dataset values blue in each line
        line2[5].set_color(BLUE)   # first x value
        line2[9].set_color(BLUE)   # first y value
        for line in [line3, line4]:
            line[3].set_color(BLUE)   # x value
            line[7].set_color(BLUE)   # y value
        
        # Group and position all lines
        lines = VGroup(line1, line2, line3, line4)
        lines.arrange(DOWN)
        lines.next_to(data_points, DOWN, buff=1)
        
        # Display each line sequentially
        for line in lines:
            self.play(Write(line))
        self.wait()

        # Indicate the x in the NN equation
        self.play(Indicate(nn_eq[6]))
        self.wait(0.5)

        # Indicate all dataset values simultaneously
        data_anims = []
        for i, line in enumerate(lines):
            if i == 1: 
                data_anims.extend([Indicate(line[5]), Indicate(line[9])])
            elif i > 1:
                data_anims.extend([Indicate(line[3]), Indicate(line[7])])
        self.play(*data_anims)
        self.wait()

                # Indicate the last two parameters in the NN equation
        self.play(Indicate(nn_eq[5]), Indicate(nn_eq[8]))
        self.wait(0.5)

        # Indicate all parameter values (a,b) simultaneously
        param_anims = []
        for i, line in enumerate(lines):
            if i == 0:  # first line has a,b in positions 1,3
                param_anims.extend([Indicate(line[1]), Indicate(line[3])])
            elif i == 1:  # second line has a,b in positions 3,7
                param_anims.extend([Indicate(line[3]), Indicate(line[7])])
            elif i > 1:   # other lines have a,b in positions 1,5
                param_anims.extend([Indicate(line[1]), Indicate(line[5])])
        self.play(*param_anims)
        self.wait()


class LossLandscape(ThreeDScene):
    def construct(self):   
        range = 5
        scale = 0.7

        # Create 3D axes
        axes = ThreeDAxes().scale(scale)
        # Create base plane
        plane = Surface(
            lambda u, v: np.array([u, v, 0]),
            u_range=[-range, range],
            v_range=[-range, range],
            resolution=(20, 20),
            fill_color=None,
            fill_opacity=0,
            stroke_width=1,
            stroke_color=WHITE,
            stroke_opacity=0.3
        ).scale(scale)

        # Set initial camera position
        self.set_camera_orientation(phi=60*DEGREES, theta=45*DEGREES)

        self.begin_ambient_camera_rotation(rate=-0.15)
        self.play(Create(plane))
        self.wait()

        # Create points for each grid position
        points = VGroup()
        steps = 11
        step = range*2/(steps-1)
        a_range = np.arange(-range, range+.01, step)
        b_range = np.arange(-range, range+.01, step)
        print(a_range, b_range)

        total_points = len(a_range) * len(b_range)
        print(total_points)

        param_point = Dot3D(
            point=axes.c2p(-range, -range, 0),
            color=GREEN,
            radius=0.05,
            resolution=(3, 3)
        )
        self.add(param_point)
        maximum = 0
        minimum = 10
        for a_val in a_range:
            for b_val in b_range:
                print(a_val, b_val)
                loss = loss_landscape(a_val, b_val)
                if loss > maximum:
                    maximum = loss
                if loss < minimum:
                    minimum = loss
                point = Dot3D(
                    point=axes.c2p(a_val, b_val, loss),
                    color=RED if loss > 0.5 else DARK_BLUE,
                    radius=0.06,
                    resolution=(2, 2)
                )
                points.add(point)
                self.play(Create(point), 
                          param_point.animate.move_to(axes.c2p(a_val, b_val, 0)),
                          run_time=0.1, 
                          rate_func=linear)
        print(maximum, minimum)
        self.wait(3)


class LocalSearch(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()

        # Create the loss surface
        landscape = Surface(
            lambda u, v: axes.c2p(u, v, loss_landscape(u, v)),
            u_range=[-5, 5],
            v_range=[-5, 5],
            resolution=(20, 20),
            checkerboard_colors=[RED_C, RED_E],
            fill_opacity=1
        )
        axes.add(landscape)
        axes.scale(0.9)
        lift = 0.15
        # Create a dot that will move along the curve
        best_point = Dot3D(
            point=axes.c2p(-1.5, -0.5, loss_landscape(-1.5, -0.5)+lift),
            color=GREEN,
            radius=0.1,
            resolution=(8, 8)
        )
        
        self.set_camera_orientation(phi=60*DEGREES, theta=45*DEGREES)

        # Add graph and dot to scene
        self.play(Create(landscape))
        self.play(Create(best_point))
        self.wait()

        mutation_rate = 0.8
        population = 5
        def create_child(dot):
            return Dot3D(
                point=np.array([dot.get_x(), dot.get_y(), dot.get_z()]),
                color=GREEN,
                radius=dot.get_radius(),
                resolution=(8, 8)
            )

        def mutate(dot):
            point = axes.p2c(dot.get_center())
            def mutate_val(val):
                return val + np.random.uniform(-mutation_rate, mutation_rate)
            x = mutate_val(point[0])
            y = mutate_val(point[1])
            z = loss_landscape(x, y)+lift
            return dot.animate.move_to(axes.c2p(x, y, z))
        
        first_child = create_child(best_point)
        self.add(first_child)
        self.play(mutate(first_child))
        self.wait()

        anims = []
        children = [best_point, first_child]
        for i in range(population-1):
            child = create_child(best_point)
            children.append(child)
            self.add(child)
            anims.append(mutate(child))
        self.play(*anims)
        self.wait()

        for child in children:
            if child.get_z() < best_point.get_z():
                best_point = child
        remove_anims = []
        for child in children:
            if child != best_point:
                remove_anims.append(FadeOut(child))
        self.play(*remove_anims)
        self.wait()

        for _ in range(6):
            anims = []
            children = [best_point]
            for i in range(population):
                child = create_child(best_point)
                children.append(child)
                self.add(child)
                anims.append(mutate(child))
            self.play(*anims, run_time=0.5)
            for child in children:
                if child.get_z() < best_point.get_z():
                    best_point = child
            remove_anims = []
            for child in children:
                if child != best_point:
                    remove_anims.append(FadeOut(child))
            self.play(*remove_anims, run_time=0.5)
        self.wait(2)

        
class CalculateGradient(Scene):
    def construct(self):
        # First show the loss landscape function
        loss_func = VGroup(
            MathTex("\\text{LossLandscape}(a,b)="),
            MathTex("\\frac{1}{3}", "\\Big(", "\\big(\\tanh(ax_1+b)-y_1\\big)^2", "+"),
            MathTex("\\quad\\big(\\tanh(ax_2+b)-y_2\\big)^2", "+"),
            MathTex("\\quad\\big(\\tanh(ax_3+b)-y_3\\big)^2", "\\Big)")
        )
        loss_func.arrange(DOWN)
        
        self.play(Write(loss_func))
        self.wait(1)

        # Show gradient definition as a vector with more vertical spacing
        gradient_def = MathTex(
            "\\nabla L(a,b) = \\begin{bmatrix} "
            "\\frac{\\partial L}{\\partial a}(a,b) \\\\ \\\\"  # Added more line breaks
            "\\frac{\\partial L}{\\partial b}(a,b)"
            "\\end{bmatrix}"
        ).scale(1.2).shift(UP)
        
        self.play(
            FadeOut(loss_func),
            FadeIn(gradient_def)
        )
        self.wait(5)

        # Show expanded partial derivatives in vector form
        expanded_gradient = MathTex(
            "\\begin{bmatrix} "
            "\\frac{\\partial L}{\\partial a}(a,b) \\\\ \\\\"  # Added more line breaks to match
            "\\frac{\\partial L}{\\partial b}(a,b)"
            "\\end{bmatrix}",
            "=",
            "\\begin{bmatrix} "
            "\\frac{1}{3}\\sum_{i=1}^3 2(\\tanh(ax_i+b)-y_i)x_i\\text{sech}^2(ax_i+b) \\\\ \\\\"
            "\\frac{1}{3}\\sum_{i=1}^3 2(\\tanh(ax_i+b)-y_i)\\text{sech}^2(ax_i+b)"
            "\\end{bmatrix}"
        ).scale(0.9)
        
        # Position the expanded gradient below the definition
        expanded_gradient.next_to(gradient_def, DOWN, buff=1)
        
        # Transform the gradient definition into the expanded form
        self.play(
            Transform(
                gradient_def.copy(),
                expanded_gradient[0]
            ),
            Transform(
                gradient_def.copy(),
                expanded_gradient[1:]
            )
        )
        self.wait(5)

class GradientDescent(ThreeDScene):
    def construct(self):
        # Create axes and landscape (same as LocalSearch)
        axes = ThreeDAxes()
        landscape = Surface(
            lambda u, v: axes.c2p(u, v, loss_landscape(u, v)),
            u_range=[-5, 5],
            v_range=[-5, 5],
            resolution=(20, 20),
            checkerboard_colors=[RED_C, RED_E],
            fill_opacity=1
        )
        axes.add(landscape)
        axes.scale(0.9)
        
        # Set initial camera position
        self.set_camera_orientation(phi=60*DEGREES, theta=45*DEGREES)
        
        # Starting point and lift value
        start_a, start_b = -1.6, -1.5
        lift = 0.15
        
        # Create starting point dot
        current_point = Dot3D(
            point=axes.c2p(start_a, start_b, loss_landscape(start_a, start_b)+lift),
            color=GREEN,
            radius=0.1,
            resolution=(8, 8)
        )
        
        # Calculate gradient at starting point
        grad_a = dL_da(start_a, start_b)
        grad_b = dL_db(start_a, start_b)
        
        # Create gradient vector display - add this AFTER setting camera orientation
        gradient_text = MathTex(
            "\\begin{bmatrix} "
            f"{grad_a:.2f} \\\\ "
            f"{grad_b:.2f}"
            "\\end{bmatrix}"
        ).to_corner(UR).set_stroke(BLACK, 2, background=True)
        
        # Add text as fixed in frame
        self.add_fixed_in_frame_mobjects(gradient_text)
        
        # Create gradient arrow
        arrow_scale = 0.5
        arrow = Arrow3D(
            start=current_point.get_center(),
            end=current_point.get_center() + np.array([
                grad_a * arrow_scale,
                grad_b * arrow_scale,
                (grad_a * arrow_scale * dL_da(start_a, start_b) + grad_b * arrow_scale * dL_db(start_a, start_b))
            ]),
            color=GREEN
        )
        
        # Add everything to scene
        self.add(landscape)
        self.play(
            Create(current_point),
            Write(gradient_text)
        )
        self.wait()
        
        # Show positive gradient arrow
        self.play(Create(arrow))
        self.wait(3)
        
        # Flip arrow to show negative gradient (direction we want to move)
        negative_arrow = Arrow3D(
            start=current_point.get_center(),
            end=current_point.get_center() + np.array([
                -grad_a * arrow_scale,
                -grad_b * arrow_scale,
                -(grad_a * arrow_scale * dL_da(start_a, start_b) + grad_b * arrow_scale * dL_db(start_a, start_b))
            ]),
            color=GREEN
        )
        
        # Update gradient display with negative sign
        negative_text = MathTex("-").next_to(gradient_text, LEFT).set_stroke(BLACK, 2, background=True)
        
        # Add new text as fixed in frame
        self.add_fixed_in_frame_mobjects(negative_text)
        
        self.play(
            ReplacementTransform(arrow, negative_arrow),
            Create(negative_text)
        )
        self.wait(2)

        param_text = MathTex(
            "\\begin{bmatrix} "
            f"{start_a:.1f} \\\\ "
            f"{start_b:.1f}"
            "\\end{bmatrix}"
        ).set_stroke(BLACK, 2, background=True)
        self.add_fixed_in_frame_mobjects(param_text)
        param_text.next_to(negative_text, LEFT)

        next_a = start_a - grad_a
        next_b = start_b - grad_b
        self.play(
            FadeOut(negative_arrow),
            current_point.animate.move_to(axes.c2p(next_a, next_b, loss_landscape(next_a, next_b)+lift)),
            Write(param_text)
        )
        self.wait()

        # return to original point
        self.play(current_point.animate.move_to(axes.c2p(start_a, start_b, loss_landscape(start_a, start_b)+lift)))
        self.wait()

        
        lr = 0.2
        scaled_grad_text = MathTex("\\begin{bmatrix} "
            f"{grad_a * lr:.2f} \\\\ "
            f"{grad_b * lr:.2f}"
            "\\end{bmatrix}"
        ).set_stroke(BLACK, 2, background=True)
        self.add_fixed_in_frame_mobjects(scaled_grad_text)
        scaled_grad_text.move_to(gradient_text)
        next_a = start_a - grad_a * lr
        next_b = start_b - grad_b * lr
        self.remove(gradient_text)
        self.play(
            current_point.animate.move_to(axes.c2p(next_a, next_b, loss_landscape(next_a, next_b)+lift)),
            Transform(gradient_text, scaled_grad_text)
        )
        self.wait()
