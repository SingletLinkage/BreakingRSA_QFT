from manim import *
import numpy as np

class FourierSeparation(Scene):
    def construct(self):
        # Parameters
        freq1 = 2  # Hz
        freq2 = 3  # Hz
        amp1 = 1
        amp2 = 1
        t_max = 4
        n_points = 1000
        t = np.linspace(0, t_max, n_points)
        
        # Signals
        y1 = amp1 * np.sin(2 * np.pi * freq1 * t)
        y2 = amp2 * np.sin(2 * np.pi * freq2 * t)
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
            x_values=t, y_values=mixed, add_vertex_dots=False, line_color="#8CD9B3"  # Light green
        )
        mixed_label = Text("2 Hz + 3 Hz", font_size=36, color=WHITE).next_to(axes, UP, buff=0.1)
        
        # Create labels
        intensity_label = Text("Intensity", font_size=24, color=WHITE).next_to(axes, LEFT, buff=0.5)
        time_label = Text("Time", font_size=24, color=WHITE).next_to(axes, RIGHT, buff=0.5)
        
        # Show the mixed signal
        self.play(Create(axes), Write(mixed_label), Write(intensity_label), Write(time_label))
        self.play(Create(mixed_graph))
        self.wait(1)
        
        # Create frequency domain graph
        freq_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 1, 0.5],
            axis_config={"color": WHITE},
            tips=False,
            x_length=6,
            y_length=3,
        ).to_edge(RIGHT).shift(DOWN * 1.5)
        
        freq_title = Text("Frequency Response", font_size=24, color="#E57373").next_to(freq_axes, UP, buff=0.2)
        freq_x_label = Text("Frequency (Hz)", font_size=20, color=WHITE).next_to(freq_axes, DOWN, buff=0.3)
        
        # Create grid for the wrapping circle
        grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        ).scale(0.8).to_edge(LEFT).shift(DOWN * 1.5)
        
        self.play(
            Create(grid),
            Create(freq_axes),
            Write(freq_title),
            Write(freq_x_label)
        )
        
        # Create a tracked dot for the signal
        track_dot = Dot(color=RED)
        track_dot.move_to(axes.c2p(0, mixed[0]))
        
        # Create frequency response peaks (initially empty)
        freq_peaks = VGroup()
        
        # Test different frequencies to see which ones resonate
        test_frequencies = [1.0, 2.0, 3.0, 4.0]
        
        for i, test_freq in enumerate(test_frequencies):
            # Title for current frequency test
            freq_test_title = Text(f"Testing {test_freq} Hz", font_size=28, color="#5CD9EF")
            freq_test_title.to_edge(UP+LEFT).shift(DOWN * 0.1)
            
            if i == 0:
                self.play(Create(freq_test_title))
            else:
                self.play(Transform(freq_test_title, Text(f"Testing {test_freq} Hz", font_size=28, color="#5CD9EF").to_edge(UP+LEFT).shift(DOWN * 0.1)))
            
            # Create wrapping circle
            wrap_circle_radius = 2
            circle = Circle(radius=wrap_circle_radius, color=GREEN_C).move_to(grid.get_center())
            
            # Create wrapping dot
            wrap_dot = Dot(color=RED)
            wrap_dot.move_to(grid.get_center() + RIGHT * wrap_circle_radius)  # Start at 0 degrees
            
            # Create path for the wrapped signal
            wrap_path = VMobject(color=GREEN_C)
            wrap_path.set_points_as_corners([wrap_dot.get_center()])
            
            # Center of mass tracking
            com_dot = Dot(color=YELLOW)
            com_dot.move_to(grid.get_center())
            
            # Vector from circle center to wrapping dot
            vector = Arrow(grid.get_center(), wrap_dot.get_center(), buff=0, color=RED, stroke_width=2)
            
            # Show initial objects
            self.play(
                Create(circle), 
                Create(track_dot), 
                Create(wrap_dot),
                Create(vector),
                Create(com_dot)
            )
            
            # Values to track during animation
            accumulated_x = 0
            accumulated_y = 0
            points_count = 0
            com_coords = []
            
            # Custom animation to wrap the signal
            wrap_animation_duration = 5  # seconds
            dt = 1/30  # Animation time step
            curr_t = 0
            
            while curr_t < wrap_animation_duration:
                # Calculate current time in the signal
                signal_t = (curr_t % t_max)
                idx = int((signal_t / t_max) * (n_points - 1))
                
                # Update position on the original signal
                track_dot.move_to(axes.c2p(signal_t, mixed[idx]))
                
                # Update position on the wrapping circle
                angle = 2 * np.pi * test_freq * signal_t
                new_pos = grid.get_center() + wrap_circle_radius * np.array([np.cos(angle), np.sin(angle), 0])
                wrap_dot.move_to(new_pos)
                
                # Update vector
                vector.put_start_and_end_on(grid.get_center(), wrap_dot.get_center())
                
                # Update path
                if len(wrap_path.get_points()) > 1000:
                    # Keep only the last 1000 points to prevent slowdown
                    old_points = wrap_path.get_points()[-1000:]
                    wrap_path.set_points_as_corners(old_points)
                
                wrap_path.add_points_as_corners([wrap_dot.get_center()])
                
                # Update center of mass calculation
                # We'll use the dot's position relative to the circle center
                rel_x = wrap_dot.get_center()[0] - grid.get_center()[0]
                rel_y = wrap_dot.get_center()[1] - grid.get_center()[1]
                
                accumulated_x += rel_x
                accumulated_y += rel_y
                points_count += 1
                
                avg_x = accumulated_x / points_count
                avg_y = accumulated_y / points_count
                
                # Only update COM every few frames to make it visible
                if curr_t % 0.1 < dt:
                    com_dot.move_to(grid.get_center() + np.array([avg_x, avg_y, 0]) * 0.2)  # Scale for visibility
                    com_coords.append((avg_x, avg_y))
                
                # Render the frame
                self.wait(dt)
                curr_t += dt
            
            # Calculate magnitude of center of mass (this represents response at this frequency)
            final_com_x = accumulated_x / points_count
            final_com_y = accumulated_y / points_count
            magnitude = np.sqrt(final_com_x**2 + final_com_y**2) / wrap_circle_radius
            
            # Create a peak in the frequency domain at this test frequency
            freq_dot = Dot(color=YELLOW)
            freq_dot.move_to(freq_axes.c2p(test_freq, magnitude))
            
            # Add a vertical line to show the peak height
            peak_line = Line(
                start=freq_axes.c2p(test_freq, 0),
                end=freq_axes.c2p(test_freq, magnitude),
                color=YELLOW
            )
            
            # Add to frequency response
            freq_peaks.add(freq_dot, peak_line)
            
            # Show the result in the frequency domain
            self.play(Create(freq_dot), Create(peak_line))
            
            # Clean up for next frequency
            self.play(
                FadeOut(wrap_path),
                FadeOut(circle),
                FadeOut(wrap_dot),
                FadeOut(vector),
                FadeOut(com_dot)
            )
            
            self.wait(0.5)
            
        # Highlight the frequency components of the original signal
        freq1_highlight = Circle(radius=0.2, color=YELLOW).move_to(freq_axes.c2p(freq1, 0.95))
        freq2_highlight = Circle(radius=0.2, color=YELLOW).move_to(freq_axes.c2p(freq2, 0.95))
        
        self.play(Create(freq1_highlight), Create(freq2_highlight))
        
        # Add explanation text
        explanation = Text("The signal contains frequencies at 2 Hz and 3 Hz", 
                          font_size=24).to_edge(DOWN)
        self.play(Write(explanation))
        
        # Explain the final result
        self.wait(2)
        
        # Transition to showing the component signals
        self.play(
            FadeOut(grid),
            FadeOut(track_dot),
            FadeOut(freq_test_title),
            FadeOut(explanation)
        )
        
        # Component signals plots
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
            x_values=t, y_values=y1, add_vertex_dots=False, line_color="#D683CE"  
        )
        
        y2_graph = axes2.plot_line_graph(
            x_values=t, y_values=y2, add_vertex_dots=False, line_color="#E5EE4F"  
        )
        
        y1_label = Text("2 Hz", font_size=28, color="#D683CE").next_to(axes1, UP, buff=0.2)
        y2_label = Text("3 Hz", font_size=28, color="#E5EE4F").next_to(axes2, UP, buff=0.2)
        
        # Add arrows and text
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
        
        ft_text = Text("Fourier decomposition", font_size=26).move_to(
            (arrow1.get_center() + arrow2.get_center()) / 2
        )
        
        # Show the separation
        self.play(
            Create(arrow1), Create(arrow2), Write(ft_text),
            FadeOut(freq_peaks), FadeOut(freq1_highlight), FadeOut(freq2_highlight)
        )
        
        self.play(
            Create(axes1), Create(axes2),
            Write(y1_label), Write(y2_label)
        )
        
        self.play(
            Create(y1_graph), Create(y2_graph)
        )
        
        self.wait(2)
