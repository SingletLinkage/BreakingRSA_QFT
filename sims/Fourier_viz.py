from manim import *
import numpy as np

class FourierSeparation(Scene):
    def construct(self):
        # Parameters
        freq1 = 2  # Hz
        freq2 = 3  # Hz
        amp1 = 1
        amp2 = 0.7
        t_max = 4
        n_points = 1000
        t = np.linspace(0, t_max, n_points)

        # Signals
        y1 = amp1 * np.sin(2 * np.pi * freq1 * t) 
        y2 = amp2 * np.sin(2 * np.pi * freq2 * t) 
        # Mixed signal
        mixed = y1 + y2

        # Main axes (mixed signal)
        axes = Axes(
            x_range=[0, t_max, 1],
            y_range=[-2, 2, 1],
            axis_config={"color": WHITE},
            tips=False,
            x_length=10,
            y_length=3,
        ).to_edge(UP, buff=0.5)

        # Mixed signal plot
        mixed_graph = axes.plot_line_graph(
            x_values=t, y_values=mixed, add_vertex_dots=False, line_color=YELLOW
        )
        mixed_label = Text("Mixed Signal", font_size=28).next_to(axes, UP, buff=0.1)

        # Pure signals plots (improved positioning)
        axes1 = Axes(
            x_range=[0, t_max, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"color": WHITE},
            tips=False,
            x_length=5,
            y_length=2.5,
        ).to_edge(LEFT).shift(DOWN * 1.5)
        
        axes2 = Axes(
            x_range=[0, t_max, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"color": WHITE},
            tips=False,
            x_length=5,
            y_length=2.5,
        ).to_edge(RIGHT).shift(DOWN * 1.5)
        
        y1_graph = axes1.plot_line_graph(
            x_values=t, y_values=y1, add_vertex_dots=False, line_color=BLUE
        )
        y2_graph = axes2.plot_line_graph(
            x_values=t, y_values=y2, add_vertex_dots=False, line_color=GREEN
        )
        
        y1_label = Text("Frequency 2 Hz", font_size=24, color=BLUE).next_to(axes1, DOWN, buff=0.2)
        y2_label = Text("Frequency 3 Hz", font_size=24, color=GREEN).next_to(axes2, DOWN, buff=0.2)

        # Improved Fourier transform arrows
        arrow1 = Arrow(
            start=axes.get_bottom() + LEFT * 2, 
            end=axes1.get_top(), 
            buff=0.2, 
            color=WHITE,
            stroke_width=2
        )
        
        arrow2 = Arrow(
            start=axes.get_bottom() + RIGHT * 2, 
            end=axes2.get_top(), 
            buff=0.2, 
            color=WHITE,
            stroke_width=2
        )
        
        ft_text = Text("Fourier Transform", font_size=26).move_to(
            (arrow1.get_center() + arrow2.get_center()) / 2 + UP * 0.2
        )

        # Animation
        self.play(Create(axes), Write(mixed_label))
        self.play(Create(mixed_graph))
        self.wait(1)
        self.play(
            Create(arrow1), Create(arrow2), FadeIn(ft_text)
        )
        self.wait(0.5)
        self.play(
            Create(axes1), Create(axes2),
            FadeIn(y1_label), FadeIn(y2_label)
        )
        self.wait(0.5)
        self.play(
            Create(y1_graph), Create(y2_graph)
        )
        self.wait(2)