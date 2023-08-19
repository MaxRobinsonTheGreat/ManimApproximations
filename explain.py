from manim import *
import numpy as np
import math

def sci_not(x):
    if x == 0:
        return "0"
    significant = x / (10**int(math.log10(abs(x))))
    exponent = int(math.log10(abs(x)))
    return "{0:.2f} \\times 10^{{{1}}}".format(significant, exponent)

def CreateNN(in_size, out_size, hidden_layers, hidden_size):
    layer_sizes = [in_size] + [hidden_size]*hidden_layers + [out_size]

    layers = []
    prev_layer = None
    for i, size in enumerate(layer_sizes):
        layer = VGroup(*[Circle(color=PINK, fill_color=LIGHT_PINK, fill_opacity=1, radius=0.15) for _ in range(size)])
        layer.arrange(DOWN, buff=0.7)
        if prev_layer is not None:
            layer.next_to(prev_layer, RIGHT, buff=1.5)
        layers.append(layer)
        prev_layer = layer

    neurons = VGroup(*layers)

    lines = VGroup()
    for i in range(len(layers) - 1):
        for neuron1 in layers[i]:
            for neuron2 in layers[i+1]:
                line = Line(neuron1.get_center(), neuron2.get_center(), color=PINK, )
                line.z_index = -1
                lines.add(line)

    network = VGroup(neurons, lines).center()
    
    return network


class Functions(Scene):
    def construct(self):
        func = MathTex(r"f(x)").scale(1.5)
        funcbox = RoundedRectangle(width=3, height=2, corner_radius=0.3, color=PURPLE_E, fill_color=PURPLE, fill_opacity=1)
        funcbox.z_index = -1
        self.play(Write(func), Create(funcbox))
        self.wait(2)

        inarrow = MathTex(r"\rightarrow").scale(1.5).next_to(funcbox, LEFT)
        inputs = MathTex(r"x").scale(1.5).next_to(inarrow, LEFT)
        outarrow = MathTex(r"\rightarrow").scale(1.5).next_to(funcbox, RIGHT)
        outputs = MathTex(r"y").scale(1.5).next_to(outarrow, RIGHT)
        self.play(Write(inarrow), Write(inputs))
        self.play(Write(outarrow), Write(outputs))
        self.wait(3)

        sinewave = MathTex(r"y = \sin(x)").scale(1.5).next_to(funcbox, DOWN*2)
        self.play(Write(sinewave))
        self.wait(3)


class NeuralArchitecture(Scene):
    def construct(self):
        approximation = MathTex(r"f(x) \approx \text{NN}(x)").scale(1.5)
        self.play(Write(approximation))
        self.wait()
        self.play(FadeOut(approximation))

        nn_function = MathTex(r"NN(x_1,x_2) = y_1, y_2")
        self.play(Write(nn_function))
        self.wait(3)

        # input vector with x_1, x_2
        input_vector = MathTex(r"\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}")
        output_vector = MathTex(r"\begin{bmatrix} y_1 \\ y_2 \end{bmatrix}")
        arrow = MathTex(r"\rightarrow").next_to(input_vector, RIGHT)
        output_vector.next_to(arrow, RIGHT)
        func_group = VGroup(input_vector, arrow, output_vector).scale(1.5).center()
        
        # replace nn_function with input_vector and output_vector
        self.play(ReplacementTransform(nn_function, func_group))
        self.wait(3)

        inputs = np.array([[0.3], [1.8]])
        input_vector = Matrix(inputs).shift(LEFT*4)
        inputs_title = Text("Inputs").scale(0.8).next_to(input_vector, UP*2)
        self.play(ReplacementTransform(func_group, input_vector), Write(inputs_title))
        self.wait(2)
        # append bias to raw_inputs
        inputs_bias = np.append(inputs, [[1]], axis=0)
        input_vector_bias = Matrix(inputs_bias).shift(LEFT*4)
        self.play(ReplacementTransform(input_vector, input_vector_bias))

        # define weights
        weights = np.array([[0.2], [-0.4], [0.6]])
        weights_vector = Matrix(weights)
        
        multiplier_operator = MathTex(r"\cdot").next_to(input_vector_bias, RIGHT)
        weights_vector.next_to(multiplier_operator, RIGHT)
        weights_title = Text("Weights").scale(0.8).next_to(weights_vector, UP)

        self.play(Write(weights_vector), Write(multiplier_operator), Write(weights_title))
        self.wait(2)
        
        # computer pointwise multiplication
        pointwise_product = np.round(inputs_bias * weights, 2)
        equal_sign = MathTex(r"\rightarrow").next_to(weights_vector, RIGHT)
        pointwise_product_vector = Matrix(pointwise_product).next_to(equal_sign, RIGHT)
        self.play(Write(equal_sign), Write(pointwise_product_vector))
        self.wait(2)

        # sum up the pointwise product and replace the pointwise product vector with the sum
        sum = str(np.round(np.sum(pointwise_product), 2))
        sum_text = MathTex(sum).next_to(equal_sign, RIGHT)
        self.play(ReplacementTransform(pointwise_product_vector, sum_text))
        self.wait(2)
        # replace sum value with ReLU(sum) text
        relu_text = MathTex(r"\text{ReLU}(" + sum + ")").next_to(equal_sign, RIGHT)
        self.play(ReplacementTransform(sum_text, relu_text))
        self.wait(2)
        
        def relu(x):
            return np.maximum(0, x)
        relu_value = MathTex(str(relu(np.sum(pointwise_product)))).next_to(equal_sign, RIGHT)
        self.play(ReplacementTransform(relu_text, relu_value))
        self.wait(2)

        # remove the weights, sum, equal sign, and activation value
        # replace with a matrix of weights for 3 neurons, including original weights
        self.remove(equal_sign, relu_value)
        weights = weights.flatten()
        weights = np.array([weights, [-0.1, 0.2, 0.1], [0.3, 1, -0.2]])
        weights_matrix = Matrix(weights).next_to(multiplier_operator, RIGHT)
        self.play(ReplacementTransform(weights_vector, weights_matrix.get_brackets()), weights_title.animate.next_to(weights_matrix, UP))
        rows = weights_matrix.get_rows()
        for row in rows:
            self.play(Write(row))
        self.wait(2)

        # compute the dot product of the weights and the inputs
        dot_product = np.round(np.dot(weights, inputs_bias), 2)
        equal_sign.next_to(weights_matrix, RIGHT)
        dot_product_vector = Matrix(dot_product).next_to(equal_sign, RIGHT)
        self.play(Write(equal_sign), Write(dot_product_vector))
        self.wait(2)

        # apply ReLU to the dot product vector (don't animate it)
        relu_vector = relu(dot_product)
        relu_vector = Matrix(relu_vector).next_to(equal_sign, RIGHT)
        self.play(ReplacementTransform(dot_product_vector, relu_vector))
        self.wait(3)

        # fade out the input vector and the weights matrix, move output to center

        self.play(FadeOut(weights_matrix), FadeOut(multiplier_operator), FadeOut(equal_sign), FadeOut(inputs_title), FadeOut(weights_title))
        
        arrow = MathTex(r"\rightarrow")
        inputs = np.array([[0.3], [1.8]])
        input_vector = Matrix(inputs).shift(LEFT*4)
        input_vector.next_to(arrow, LEFT)
        self.play(Write(arrow), ReplacementTransform(input_vector_bias, input_vector), relu_vector.animate.next_to(arrow, RIGHT))
        layers_group = VGroup(input_vector, arrow, relu_vector)
        self.wait(2)

        outputs_list = [[[0.3], [1.8], [2.4], [0.0]],
                        [[3.5], [0.0], [0.4]],
                        [[2.9], [1.1]]]
        outputs_vector = None
        for outputs in outputs_list:
            arrow = MathTex(r"\rightarrow").next_to(layers_group, RIGHT)
            outputs_vector = Matrix(outputs).next_to(arrow, RIGHT)
            layers_group.add(arrow, outputs_vector)
            self.play(FadeIn(arrow), FadeIn(outputs_vector), layers_group.animate.shift(LEFT*1.4))
            self.wait(0.5)
        self.play(Indicate(outputs_vector))
        self.wait(3)


class NeuralArchitecture2(Scene):
    def construct(self):
        # fully connected nn equation
        nn_equation = MathTex(r"\text{NN}(x) = \text{ReLU}(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot x)))").shift((UP*2))
        self.play(Write(nn_equation))
        self.wait(2)

        # fully connected nn diagram as equal to nn equation under it
        equal = MathTex(r"=").next_to(nn_equation, DOWN, buff=1).scale(1.5)
        nn = CreateNN(1, 1, 2, 3).next_to(equal, DOWN, buff=1).scale(1.2)
        self.play(Write(nn), Write(equal))
        self.wait(2)

        self.remove(equal, nn_equation)
        self.play(nn.animate.center())

        nn2 = CreateNN(1, 1, 2, 4).scale(1.2)
        self.play(ReplacementTransform(nn, nn2))
        self.wait()

        nn3 = CreateNN(1, 1, 3, 4).scale(1.2)
        self.play(ReplacementTransform(nn2, nn3))
        self.wait()

        nn4 = CreateNN(1, 1, 4, 5).scale(1.2)
        self.play(ReplacementTransform(nn3, nn4))
        self.wait()

        nn5 = CreateNN(1, 1, 5, 6).scale(1.2)
        self.play(ReplacementTransform(nn4, nn5))
        self.wait(3)


class NeuralArchitecture3(Scene):
    def construct(self):
        approximation = MathTex(r"f(x) \approx \text{NN}(x)").scale(1.5)
        self.play(Write(approximation))
        self.wait()
        self.play(FadeOut(approximation))

        nn = CreateNN(3, 3, 2, 4).scale(1.2)
        self.play(Write(nn))
        self.wait(2)

        # input vector with x_1, x_2
        input_vector = MathTex(r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}").scale(1.5).next_to(nn, LEFT)
        output_vector = MathTex(r"\begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix}").scale(1.5).next_to(nn, RIGHT)
        self.play(Write(input_vector), Write(output_vector))
        self.wait(3)

        neuron = CreateNN(3, 1, 1, 1).scale(1.2)
        self.play(ReplacementTransform(nn, neuron), FadeOut(output_vector), input_vector.animate.next_to(neuron, LEFT))
        self.wait(2)

        # weighted sum tex of 3vector
        weighted_sum = MathTex(r"= w_1x_1 + w_2x_2 + w_3x_3 + w_4").next_to(neuron, DOWN).scale(1.5)
        self.play(Write(weighted_sum))
        self.wait()

        self.play(FadeOut(neuron), FadeOut(input_vector), weighted_sum.animate.center())
        self.wait()

        input_vector = MathTex(r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ 1 \end{bmatrix}").scale(1.5).shift(LEFT)
        dot_product = MathTex(r"\cdot").next_to(input_vector, RIGHT)
        weights_vector = MathTex(r"\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4\end{bmatrix}").scale(1.5).next_to(dot_product, RIGHT)
        self.play(ReplacementTransform(weighted_sum.copy(), input_vector), ReplacementTransform(weighted_sum, weights_vector), Write(dot_product))
        self.wait(2)

        # replace with actual values
        inputs = np.array([[0.3], [1.8], [0.5], [1.0]])
        input_vector_real = Matrix(inputs).move_to(input_vector)
        weights = np.array([[0.2], [-0.4], [0.6], [0.1]])
        weights_vector_real = Matrix(weights).move_to(weights_vector)
        self.play(ReplacementTransform(input_vector, input_vector_real), ReplacementTransform(weights_vector, weights_vector_real))
        self.wait(2)

        inputs_vector_real_copy = input_vector_real.copy()
        sum_vector_real = Matrix(np.round(inputs * weights, 2)).move_to(weights_vector_real)
        self.play(inputs_vector_real_copy.animate.move_to(weights_vector_real), ReplacementTransform(inputs_vector_real_copy, sum_vector_real))
        self.wait(0.2)
        equal_sign = MathTex(r"=").next_to(weights_vector_real, RIGHT)
        self.play(sum_vector_real.animate.next_to(equal_sign, RIGHT), Write(equal_sign))
        self.wait()

        dot_prod = round(np.dot(inputs.flatten(), weights.flatten()), 2)
        dot_prod_real = MathTex(str(dot_prod)).next_to(equal_sign, RIGHT)
        self.play(ReplacementTransform(sum_vector_real, dot_prod_real))
        self.wait(2)

        relu = MathTex(r"\text{ReLU}(" + str(dot_prod) + ")").center()
        self.play(FadeOut(input_vector_real), FadeOut(weights_vector_real), FadeOut(dot_product), FadeOut(equal_sign), ReplacementTransform(dot_prod_real, relu))
        self.wait(2)

        # replace with actual values
        final_output = MathTex(str(round(max(dot_prod, 0),2))).center()
        self.play(ReplacementTransform(relu, final_output))
        self.wait(2)

        self.remove(final_output)

        nn_layer = CreateNN(3, 4, 0, 0).scale(1.)
        inputs = np.array([[0.3], [1.8], [0.5]])
        outputs = np.array([[0.0], [2.9], [3.7], [0.0]])
        input_vector = Matrix(inputs).scale(1.).next_to(nn_layer, LEFT, buff=0.3)
        output_vector = Matrix(outputs).scale(1.).next_to(nn_layer, RIGHT, buff=0.3)
        self.play(Write(input_vector), Create(nn_layer), Write(output_vector))
        layers_group = VGroup(input_vector, nn_layer, output_vector)
        self.wait(2)

        outputs_list = [[[0.1], [2.5], [3.2], [1.1]],
                        [[3.5], [0.0], [0.4], [3.1]],
                        [[2.9], [1.1], [0.0]]]
        outputs_vector = None
        in_size = 4
        for outputs in outputs_list:
            out_size = len(outputs)
            layer = CreateNN(in_size, out_size, 0, 0).scale(1.).next_to(layers_group, RIGHT, buff=0.3)
            outputs_vector = Matrix(outputs).next_to(layer, RIGHT, buff=0.3)
            layers_group.add(layer, outputs_vector)
            self.play(Create(layer), FadeIn(outputs_vector), layers_group.animate.shift(LEFT*3))
            self.wait(0.5)
        self.play(Indicate(outputs_vector))
        self.wait(3)


class NN(Scene):
    def construct(self):
        nn = CreateNN(3, 3, 2, 4).scale(1.5)
        self.add(nn)


class CombiningNeurons(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-5, 5, 0.5],
            y_range=[-2, 3, 0.5],
            x_length=20,
            y_length=10,
            tips=False
        ).shift(UP*2).scale(0.9)
        self.add(axes)
        def relu(x):
            return np.maximum(0, x)
        x_range = [-5, 5, 0.01]
        line = axes.plot(lambda x: relu(x), color=BLUE, use_smoothing=False)
        equ = MathTex(r"\text{ReLU}(x)").next_to(axes, DOWN).scale(0.9)
        self.play(Write(line), Write(equ))
        self.wait(1)
        line0 = axes.plot(lambda x: relu(-x+2), color=BLUE, use_smoothing=False, x_range=x_range)
        equ0 = MathTex(r"\text{ReLU}(-x+2)").next_to(axes, DOWN).scale(0.9)
        self.play(ReplacementTransform(line, line0), ReplacementTransform(equ, equ0))
        self.wait(1)
        line1 = axes.plot(lambda x: relu(3*x-1), color=BLUE, use_smoothing=False, x_range=x_range)
        equ1 = MathTex(r"\text{ReLU}(3x-1)").next_to(axes, DOWN).scale(0.9)
        self.play(ReplacementTransform(line0, line1), ReplacementTransform(equ0, equ1))
        self.wait(3)
        line2 = axes.plot(lambda x: relu(3*x-1) - relu(2*x+1), color=BLUE, use_smoothing=False, x_range=x_range)
        equ2 = MathTex(r"\text{ReLU}(3x-1) - \text{ReLU}(2x+1)").next_to(axes, DOWN).scale(0.9)
        self.play(ReplacementTransform(line1, line2), ReplacementTransform(equ1, equ2))
        self.wait(0.5)
        line3 = axes.plot(lambda x: relu(3*x-1) - relu(2*x+1) + 2*relu(x+1), color=BLUE, use_smoothing=False, x_range=x_range)
        equ3 = MathTex(r"\text{ReLU}(3x-1) - \text{ReLU}(2x+1) + 2\text{ReLU}(x+1)").next_to(axes, DOWN).scale(0.9)
        self.play(ReplacementTransform(line2, line3), ReplacementTransform(equ2, equ3))
        self.wait(0.5)
        line4 = axes.plot(lambda x: relu(3*x-1) - relu(2*x+1) + 2*relu(x+1) - 0.1*relu(10*x), color=BLUE, use_smoothing=False, x_range=x_range)
        equ4 = MathTex(r"\text{ReLU}(3x-1) - \text{ReLU}(2x+1) + 2\text{ReLU}(x+1) - 0.1\text{ReLU}(10x)").next_to(axes, DOWN).scale(0.9)
        self.play(ReplacementTransform(line3, line4), ReplacementTransform(equ3, equ4))
        self.wait(3)


class ManyFunctions(Scene):
    def construct(self):
        # display 2d axes and a simple linear function
        axes = Axes(
            x_range=[-10, 10, 0.5],
            y_range=[-5, 5, 0.5],
            x_length=20,
            y_length=10,
            tips=False
        )
        axes.center()
        self.play(Write(axes))
        self.wait(0.5)

        line = axes.plot(lambda x: x, color=PURPLE)
        self.play(Write(line))
        self.wait(0.5)

        absolute = axes.plot(lambda x: abs(x), color=RED)
        self.play(ReplacementTransform(line, absolute))
        self.wait(0.5)

        cubic = axes.plot(lambda x: x**3, color=GREEN)
        self.play(ReplacementTransform(absolute, cubic))
        self.wait(0.5)
        
        log = axes.plot(lambda x: np.log(x), color=RED, x_range=[0.001, 10, 0.01])
        self.play(ReplacementTransform(cubic, log))
        self.wait(0.5)

        sine = axes.plot(lambda x: 2*np.sin(x), color=BLUE)
        self.play(ReplacementTransform(log, sine))
        self.wait(0.5)

        gaussian = axes.plot(lambda x: 2*np.exp(-x**2), color=YELLOW)
        self.play(ReplacementTransform(sine, gaussian))
        self.wait(0.5)


class Dimensionality(Scene):
    # show a vector of x_n values increasing in dimensionality and a dimensionality counter next to it
    def construct(self):
        vector = MathTex(r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}").scale(1.5).shift(LEFT)
        dimensionality = MathTex(r"\text{Dimensionality} = 3").next_to(vector, RIGHT)
        self.play(Write(vector), Write(dimensionality))
        self.wait()

        for dim in range(4,8):
            # create next vector string with as many x_n values as the dimensionality
            vector_next = MathTex(r"\begin{bmatrix} " + " \\\\ ".join([f"x_{n}" for n in range(1,dim+1)]) + " \end{bmatrix}").scale(1.5).shift(LEFT)
            dimensionality_next = MathTex(r"\text{Dimensionality} = " + str(dim)).next_to(vector_next, RIGHT)
            self.play(ReplacementTransform(vector, vector_next), ReplacementTransform(dimensionality, dimensionality_next))
            vector = vector_next
            dimensionality = dimensionality_next
            self.wait(0.5)
        self.wait(2)
        self.remove(vector, dimensionality)

        # image pixel coords to pixel value
        vectors = MathTex(r"\begin{bmatrix} row \\ col \end{bmatrix} \rightarrow \begin{bmatrix} pixel \end{bmatrix}").shift(UP).scale(1.5)
        r2_to_r1 = MathTex(r"\mathbb{R}^2 \rightarrow \mathbb{R}^1").scale(2).next_to(vectors, DOWN*2)
        self.play(Write(vectors))
        self.wait()
        self.play(Write(r2_to_r1))
        self.wait(3)
        self.remove(vectors, r2_to_r1)

        # parametric surface dimensionality
        vectors = MathTex(r"\begin{bmatrix} u \\ v \end{bmatrix} \rightarrow \begin{bmatrix} x \\ y \\ z \end{bmatrix}").shift(UP).scale(1.5)
        r2_to_r3 = MathTex(r"\mathbb{R}^2 \rightarrow \mathbb{R}^3").scale(2).next_to(vectors, DOWN*2)
        title = Text("Parametric Surface").next_to(vectors, UP)
        self.play(Write(vectors), Write(r2_to_r3))
        self.wait()
        self.play(Write(title))
        self.wait(3)
        self.remove(vectors, r2_to_r3, title)

        # parametric surface equation of sphere
        x_equ = MathTex(r"x = \cos(u)\sin(v)").shift(UP).scale(1.5)
        y_equ = MathTex(r"y = \sin(u)\sin(v)").next_to(x_equ, DOWN).scale(1.5)
        z_equ = MathTex(r"z = \cos(v)").next_to(y_equ, DOWN).scale(1.5)
        self.play(Write(x_equ), Write(y_equ), Write(z_equ))
        self.wait(3)


class ActivationFunctions(Scene):
    # show relu, leakyrelu, sigmoid, and tanh. remove each one after the other. show their equations below
    def construct(self):
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=15,
            y_length=5,
            tips=False
        ).shift(UP).scale(0.9)
        self.play(Write(axes))
        self.wait()

        # relu
        relu = axes.plot(lambda x: np.maximum(0, x), color=RED, use_smoothing=False)
        relu_equ = MathTex(r"\text{ReLU}(x) = \max(0, x)").next_to(axes, DOWN)
        self.play(Write(relu), Write(relu_equ))
        self.wait(3)
        self.remove(relu, relu_equ)
        self.wait()

        # leaky relu
        leakyrelu = axes.plot(lambda x: np.maximum(0.1*x, x), color=RED, use_smoothing=False)
        leakyrelu_equ = MathTex(r"\text{LeakyReLU}(x) = \max(0.1x, x)").next_to(axes, DOWN)
        self.play(Write(leakyrelu), Write(leakyrelu_equ))
        self.wait(3)
        self.remove(leakyrelu, leakyrelu_equ)
        self.wait()

        # sigmoid
        sigmoid = axes.plot(lambda x: 1/(1+np.exp(-x)), color=RED)
        sigmoid_equ = MathTex(r"\text{Sigmoid}(x) = \frac{1}{1+e^{-x}}").next_to(axes, DOWN)
        self.play(Write(sigmoid), Write(sigmoid_equ))
        self.wait(3)
        self.remove(sigmoid, sigmoid_equ)
        self.wait()

        # tanh
        tanh = axes.plot(lambda x: np.tanh(x), color=RED)
        tanh_equ = MathTex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}").next_to(axes, DOWN)
        self.play(Write(tanh), Write(tanh_equ))
        self.wait(3)
        # show tanh normalized to 0 1 and equation below current equation
        tanh_norm = axes.plot(lambda x: (np.tanh(x)+1)/2, color=GREEN)
        tanh_norm_equ = MathTex(r"\text{Normalized }\tanh(x) = \frac{\tanh(x)+1}{2}", color=GREEN).next_to(tanh_equ, DOWN)
        self.play(Write(tanh_norm), Write(tanh_norm_equ))
        self.wait(3)


class Normalization(Scene):
    # show simple linear normalization of many random points in 2d axis being linearly shifted from [0, 10] to [-1, 1]
    def construct(self):
        axes = Axes(
            x_range=[-2, 7, 0.5],
            y_range=[-2, 7, 0.5],
            x_length=15,
            y_length=15,
            tips=False
        ).scale(0.8).shift(UP)
        self.add(axes)
        self.wait()

        # generate random points
        points = np.random.uniform(low=0, high=5, size=(100,2))
        dots = VGroup(*[Dot(axes.c2p(point[0], point[1]), radius=0.07, color=RED) for point in points])
        self.play(FadeIn(dots))
        self.wait()
        # normalize points to [-1, 1]
        points = (points - 2.5) / 2.5
        dots2 = VGroup(*[Dot(axes.c2p(point[0], point[1]), radius=0.07, color=RED) for point in points])
        self.play(ReplacementTransform(dots, dots2), run_time=2)
        self.wait(3)


class TaylorSeries(Scene):
    def construct(self):
        title = Text("Taylor Series").scale(1.2).shift(UP)
        talyor_series_simple = MathTex(r"= 1 + x + x^2 + x^3 + x^4 + \cdots + x^n").next_to(title, DOWN)
        self.play(Write(title))
        self.wait()
        self.play(Write(talyor_series_simple))
        self.wait(2)
        talyor_series_simple_coef = MathTex(r"= a_0 + a_1x + a_2x^2 + a_3x^3 + a_4x^4 + \cdots + a_nx^n").next_to(title, DOWN)
        self.play(ReplacementTransform(talyor_series_simple, talyor_series_simple_coef))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(talyor_series_simple_coef))

        x_range = [-10, 10, 1]
        axes = Axes(
            x_range=x_range,
            y_range=[-2, 3, 1],
            x_length=20,
            y_length=6,
            tips=False
        ).center().shift(UP)

        def get_taylor_function(coefficients):
            order = len(coefficients) - 1
            def taylor(x):
                result = 0
                for i in range(order + 1):
                    result += x**i * coefficients[i]
                return result
            return taylor
        
        def get_taylor_function_tex(coefficients):
            order = len(coefficients) - 1
            result = ""
            for i in range(order + 1):
                if i == 0:
                    result += f"{coefficients[i]} + "
                else:
                    if (abs(coefficients[i]) < 0.01 and coefficients[i] != 0):
                        result += "{}x^{{}} + ".format(sci_not(coefficients[i]), i)
                    else:
                        result += f"{np.round(coefficients[i], 2)}x^{{{i}}} + "
                    
            tex = MathTex(result[:-3])
            return tex
        
        coefficients = [-1, 1, 1, 1, 1]
        taylor = get_taylor_function(coefficients)
        taylor_tex = get_taylor_function_tex(coefficients).next_to(axes, DOWN)
        taylor_graph = axes.plot(taylor, x_range=[-5, 5], color=BLUE)
        self.play(Write(taylor_tex), Write(axes))
        self.play(Write(taylor_graph), rate_func = linear)
        self.wait()

        coefficients = [1, 1.5, 0, -1.1, 0.3]
        taylor = get_taylor_function(coefficients)
        taylor_tex2 = get_taylor_function_tex(coefficients).next_to(axes, DOWN)
        taylor_graph2 = axes.plot(taylor, x_range=[-5, 5], color=BLUE)
        self.play(ReplacementTransform(taylor_tex, taylor_tex2), ReplacementTransform(taylor_graph, taylor_graph2), rate_func = linear)
        self.wait()

        # add target sine function
        sine = axes.plot(lambda x: np.sin(x), x_range=x_range, color=RED)
        sine.z_index = -1
        self.play(Write(sine))
        self.wait()

        sine_coefficients = [0, 1, 0, -1/6, 0, 1/120, 0, -1/5040, 0, 1/362880, 0, -1/39916800, 0, 1/6227020800, 0, -1/1307674368000, 0, 1/355687428096000, 0, -1/121645100408832000]
        #progressively add these coefficients to the taylor series
        coefficients = sine_coefficients[:4]
        taylor_tex = taylor_tex2
        i = len(coefficients)
        while len(coefficients) < len(sine_coefficients):
            coefficients.append(sine_coefficients[i])
            i += 1
            if (coefficients[-1] == 0):
                continue
            taylor = get_taylor_function(coefficients)
            taylor_tex = get_taylor_function_tex(coefficients).next_to(axes, DOWN).scale(0.8).move_to(taylor_tex).shift(LEFT*0.5)
            taylor_graph = axes.plot(taylor, x_range=x_range, color=BLUE)
            
            self.play(ReplacementTransform(taylor_graph2, taylor_graph), ReplacementTransform(taylor_tex2, taylor_tex), rate_func = linear)
            taylor_tex2 = taylor_tex
            taylor_graph2 = taylor_graph
            self.wait(0.5)

        self.wait(2)

        # equation to calculate coefficients a_n
        coefficient_equation = MathTex(r"a_n = \frac{f^{(n)}(0)}{n!}").shift(UP*3 + LEFT*4)
        self.play(Write(coefficient_equation))
        self.wait(3)


class TaylorFeatures(Scene):
    def construct(self):
        talyor_series = MathTex(r"a_0 + a_1x + a_2x^2 + a_3x^3 + a_4x^4").scale(1.5)
        self.play(Write(talyor_series))
        self.wait(2)

        # split a terms and x terms into tex vectors of their own
        x_vector = MathTex(r"\begin{bmatrix} 1 \\ x \\ x^2 \\ x^3 \\ x^4 \end{bmatrix}").shift(LEFT)
        dot_product = MathTex(r"\cdot").next_to(x_vector, RIGHT)
        a_vector = MathTex(r"\begin{bmatrix} a_0 \\ a_1 \\ a_2 \\ a_3 \\ a_4 \end{bmatrix}").next_to(dot_product, RIGHT)
        dot_group = VGroup(x_vector, dot_product, a_vector).scale(1.5).center()
        self.play(ReplacementTransform(talyor_series, dot_group))
        self.wait(2)

        # remove the coefficients and dot prod
        self.play(FadeOut(a_vector), FadeOut(dot_product))
        # replace with a neural network that takes 5 inputs and 1 output
        nn = CreateNN(5, 1, 2, 5).next_to(x_vector, RIGHT)
        self.play(Create(nn))
        self.wait(3)


class FourierSeries(Scene):
    def construct(self):
        title = Text("Fourier Series").scale(1.2).shift(UP)
        fourier_series_simple = MathTex(r"1 + sin(x) + cos(x) + sin(2x) + cos(2x) + \cdots + sin(nx) + cos(nx)").next_to(title, DOWN).scale(0.8)
        self.play(Write(title))
        self.wait()
        self.play(Write(fourier_series_simple))
        self.wait(2)
        fourier_series_simple_coef = MathTex(r"a_0 + a_1sin(x) + b_1cos(x) + a_2sin(2x) + b_2cos(2x) + \cdots + a_nsin(nx) + b_ncos(nx)").next_to(title, DOWN).scale(0.8)
        self.play(ReplacementTransform(fourier_series_simple, fourier_series_simple_coef))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(fourier_series_simple_coef))

        x_range = [-np.pi, np.pi, 0.1]
        axes = Axes(
            x_range=x_range,
            y_range=[-1, 2, 0.1],
            x_length=15,
            y_length=6,
            tips=False
        ).center().shift(UP)

        self.play(Write(axes))

        sine = axes.plot(lambda x: np.sin(x), color=BLUE)
        equation = MathTex(r"sin(1x)").next_to(axes, DOWN)
        self.play(Write(sine), Write(equation))
        self.wait(0.5)

        for i in range(2, 6):
            next_sine = axes.plot(lambda x: np.sin(i*x), color=BLUE)
            equation_next = MathTex(r"sin(" + str(i) + "x)").next_to(axes, DOWN)
            self.play(ReplacementTransform(sine, next_sine), ReplacementTransform(equation, equation_next))
            sine = next_sine
            equation = equation_next
            self.wait(0.2)
        self.wait()

        def get_fourier_function(coefficients):
            def fourier(x):
                result = coefficients[0]
                order = (len(coefficients)-1)//2
                for n in range(1, order+1):
                    i = 2*n - 1
                    result += coefficients[i] * np.sin(n*x) + coefficients[i+1] * np.cos(n*x)
                return result
            return fourier
        
        def get_fourier_function_tex(coefficients):
            result = f"{coefficients[0]} + "
            order = (len(coefficients)-1)//2
            for n in range(1, order+1):
                i = 2*n - 1
                result += f"{np.round(coefficients[i], 2)}sin({n}x) + {np.round(coefficients[i+1], 2)}cos({n}x) + "
            tex = MathTex(result[:-3])
            return tex
        
        coefficients = [1, 1, 1, 1, 1, 1, 1]
        fourier = get_fourier_function(coefficients)
        fourier_tex = get_fourier_function_tex(coefficients).next_to(axes, DOWN).scale(0.8)
        fourier_graph = axes.plot(fourier, x_range=x_range, color=BLUE)
        self.play(ReplacementTransform(equation, fourier_tex), ReplacementTransform(sine, fourier_graph), rate_func = linear)
        self.wait(2)

        # add a target function abs(x)
        target = axes.plot(lambda x: x/np.pi, x_range=x_range, color=RED)
        target.z_index = -1
        self.play(Write(target))
        self.wait()

        def sawtooth_coefficients(order):
            coefficients = []
            for n in range(1, order + 1):
                # a_n coefficients are all zero
                coefficients.append(0)
                # b_n coefficients for n > 0
                coefficients.append((-1)**(n+1)/(n))
            return 2/np.pi * np.array(coefficients)
        
        for order in range(3, 10):
            coefficients = sawtooth_coefficients(order)
            fourier = get_fourier_function(coefficients)
            fourier_tex_next = get_fourier_function_tex(coefficients).next_to(axes, DOWN).scale(0.6)
            fourier_graph_next = axes.plot(fourier, x_range=x_range, color=BLUE)
            self.play(ReplacementTransform(fourier_tex, fourier_tex_next), ReplacementTransform(fourier_graph, fourier_graph_next), run_time=0.5)
            fourier_tex = fourier_tex_next
            fourier_graph = fourier_graph_next
            self.wait(0.5)
        self.wait(3)

        self.remove(axes, fourier_tex, fourier_graph, target)

        # extend the approximation to range -4pi to 4pi and zoom out
        scaler = 0.1
        x_range = [-np.pi/scaler, np.pi/scaler]
        axes = Axes(
            x_range=x_range + [PI/4],
            y_range=x_range + [1],
            x_length=15/scaler,
            y_length=15/scaler,
            tips=False
        ).center()
        fourier_graph = axes.plot(fourier, x_range=x_range+[0.1], color=BLUE)
        self.add(axes, fourier_graph)
        self.wait(2)
        self.play(fourier_graph.animate.scale(scaler), axes.animate.scale(scaler), run_time=6, rate_func=rate_functions.ease_in_out_sine)
        self.wait(3)


class FourierFeatures(Scene):
    def construct(self):
        # remove the coefficients and dot prod
        x_vector = MathTex(r"\begin{bmatrix} 1 \\ sin(x) \\ cos(x) \\ sin(2x) \\ cos(2x) \\ sin(3x) \\ cos(3x) \end{bmatrix}").shift(LEFT).scale(1.5)
        self.play(Write(x_vector))
        self.wait(1)
        # replace with a neural network that takes 5 inputs and 1 output
        nn = CreateNN(7, 1, 2, 4).next_to(x_vector, RIGHT)
        self.play(Create(nn))
        self.wait(3)

        self.remove(nn, x_vector)
        x_vector_2d = MathTex(r"\begin{bmatrix} 1 \\ sin(x) \\ cos(x) \\ sin(2x) \\ cos(2x) \\ sin(y) \\ cos(y) \\ sin(2y) \\ cos(2y) \end{bmatrix}").shift(LEFT).scale(1.4)
        nn = CreateNN(9, 1, 2, 4).next_to(x_vector_2d, RIGHT).scale(0.9)
        self.play(Write(x_vector_2d))
        self.play(Create(nn))
        self.wait(3)


class Fourier2D(Scene):
    def construct(self):

        def decomp_tex(strs):
            decomp = VGroup(*[
                MathTex(tex) for tex in strs
            ]).scale(0.6)
            decomp.arrange(DOWN, center=True).center()
            return decomp

        prev_decomp = decomp_tex(self.generate_fourier_series(2))
        title = Text("2D Fourier Series (2 orders):").next_to(prev_decomp, UP)
        self.play(Write(prev_decomp), Write(title))
        self.wait(3)
        self.remove(title)
        for order in range(3, 8):
            decomp = decomp_tex(self.generate_fourier_series(order))
            self.play(ReplacementTransform(prev_decomp, decomp))
            prev_decomp = decomp
            self.wait()
        self.wait(3)
        self.remove(decomp)


        # num_terms = (2*order+1)**2
        term_calc_2d = MathTex(r"\text{Number of terms for 2D Fourier Series for order n} = (2n+1)^2").shift(UP)
        self.add(term_calc_2d)
        self.wait()

        term_calc_1d = MathTex(r"\text{Number of terms for 1D Fourier Series for order n} = 2n+1").next_to(term_calc_2d, DOWN)
        self.add(term_calc_1d)
        self.wait()

        # write "Curse of Dimensionality" above the 2d equation in creepy font
        curse = Text("Curse of Dimensionality", color=MAROON_E).scale(1.5).next_to(term_calc_2d, UP)
        self.play(Write(curse))
        self.wait()


    def generate_fourier_series(self, order):
        terms = ['cos', 'sin']
        result = ['f(x, y) = ']
        def length(strs):
            return sum(len(s) for s in strs)
        running = []
        count = 0
        for i in range(order + 1):
            for j in range(order + 1):
                for term1 in terms:
                    for term2 in terms:
                        if i == 0 and term1 == 'sin':
                            continue # the whole term is 0
                        if j == 0 and term2 == 'sin':
                            continue # the whole term is 0
                        if i == 0 and term1 == 'cos' and j == 0 and term2 == 'cos':
                            running.append('1') # the whole term is 1
                        elif i == 0 and term1 == 'cos' and j != 0:
                            running.append(f'{term2}({j} y)') # term1 is 1, only need term2
                        elif j == 0 and term2 == 'cos' and i != 0:
                            running.append(f'{term1}({i} x)') # term1 is 1, only need term2
                        else:
                            running.append(f'{term1}({i} x) {term2}({j} y)')
                        count += 1
                        # add a newline if the result is long enough
                        if length(running) > 80:
                            # combine running into single string
                            running.append('')
                            result.append(' + '.join(running))
                            running = []
        print("Number of terms:", count)
        result.append(' + '.join(running))
        return result


class DigitToVector(Scene):
    def construct(self):
        # 1. Load the image "digit.png".
        img = ImageMobject("./digit.png").scale(2)  # Scale to make it visible
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        # 3. Resize the image to 28x28
        image_length = 28
        size = 6
        img.stretch_to_fit_height(size)
        img.stretch_to_fit_width(size)

        self.add(img)
        self.wait()

        # overlay grid on top of image for each pixel
        grid = NumberPlane(
            x_range=[0, image_length, 1],
            y_range=[0, image_length, 1],
            x_length=size,
            y_length=size,
            tips=False,
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 3,
            }
        ).move_to(img.get_center())
        grid_outline = SurroundingRectangle(grid, color=WHITE, buff=0.0)
        self.play(Write(grid), FadeIn(grid_outline))
        self.wait(2)

        vector = MathTex(r"\begin{bmatrix} " + r" \\ ".join(["p_{{{}}}".format(i) for i in range(1, 12)]) + r" \\ \vdots \\ p_{784} \end{bmatrix}").shift(LEFT*5).scale(0.9)
        self.play(ReplacementTransform(grid, vector), FadeOut(grid_outline), FadeOut(img))
        self.wait(3)

        # one hot for digit 3
        out_vector = MathTex(r"\begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \\ 1.0 \\ 0.0 \\ 0.0 \\ 0.0 \\ 0.0 \\ 0.0 \\ 0.0 \end{bmatrix}").shift(RIGHT*5).scale(0.9)

        self.play(Write(out_vector))
        self.wait(3)

        func = MathTex(r"\rightarrow \text{NN}(\text{img}) \rightarrow").scale(1.5)
        self.play(Write(func))
        self.wait(3)

        nn = CreateNN(7, 5, 2, 7).scale(0.9)
        self.play(ReplacementTransform(func, nn))
        self.wait(3)


class AddingDimensions(Scene):
    def construct(self):
        nn = CreateNN(2, 2, 4, 5)
        self.play(Create(nn))
        self.wait(1)
        nn2 = CreateNN(3, 3, 4, 5)
        self.play(ReplacementTransform(nn, nn2))
        self.wait(1)
        nn3 = CreateNN(4, 4, 4, 5)
        self.play(ReplacementTransform(nn2, nn3))
        self.wait(1)
        nn4 = CreateNN(5, 5, 4, 5)
        self.play(ReplacementTransform(nn3, nn4))
        self.wait(3)