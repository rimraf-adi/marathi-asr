from manim import *
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import *

class AlphaTuning(Scene):
    def construct(self):
        title = Text("Tuning Language Model Weight (α)", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))
        
        # Axes for WER curve
        ax = Axes(
            x_range=[0, 1.0, 0.2],
            y_range=[10, 25, 5],
            x_length=6,
            y_length=4,
            axis_config={"include_numbers": False},
        ).move_to(DOWN*0.5)
        
        # Manually add text labels to prevent LaTeX hanging
        zero_label = Text("0.0", font_size=16).next_to(ax.c2p(0, 10), DOWN)
        one_label = Text("1.0", font_size=16).next_to(ax.c2p(1.0, 10), DOWN)
        y_max = Text("25", font_size=16).next_to(ax.c2p(0, 25), LEFT)
        y_min = Text("10", font_size=16).next_to(ax.c2p(0, 10), LEFT)
        
        x_label = Text("Alpha (α)", font_size=20).next_to(ax.x_axis, RIGHT)
        y_label = Text("WER %", font_size=20).next_to(ax.y_axis, UP)
        
        self.play(Create(ax), Write(x_label), Write(y_label))
        
        # Mock WER curve function: starts high at alpha=0, dips at alpha=0.5, rises slightly after
        def wer_func(x):
            return 22.0 - (40.0 * x) + (40.0 * x**2)
            
        curve = ax.plot(wer_func, color=BLUE)
        self.play(Create(curve))
        
        # Value tracker for slider
        alpha_tracker = ValueTracker(0.0)
        
        # Slider UI
        slider_line = Line(LEFT*2, RIGHT*2).next_to(ax, UP, buff=1.0)
        slider_dot = always_redraw(lambda: Dot(color=YELLOW).move_to(slider_line.point_from_proportion(alpha_tracker.get_value())))
        slider_label = always_redraw(lambda: Text(f"α = {alpha_tracker.get_value():.2f}", font_size=24, color=YELLOW).next_to(slider_dot, UP))
        
        self.play(Create(slider_line), Create(slider_dot), Write(slider_label))
        
        # Live dot on the WER curve
        curve_dot = always_redraw(lambda: Dot(color=RED).move_to(ax.c2p(alpha_tracker.get_value(), wer_func(alpha_tracker.get_value()))))
        wer_label = always_redraw(lambda: Text(f"WER: {wer_func(alpha_tracker.get_value()):.1f}%", font_size=20, color=RED).next_to(curve_dot, RIGHT, buff=0.2))
        
        self.play(Create(curve_dot), Write(wer_label))
        
        # Animate the slider from 0 to 1
        self.play(alpha_tracker.animate.set_value(1.0), run_time=4, rate_func=there_and_back)
        
        # Settle on optimal alpha = 0.5
        self.play(alpha_tracker.animate.set_value(0.5), run_time=2)
        
        optimal_label = Text("Optimal α", font_size=20, color=GREEN).next_to(curve_dot, DOWN)
        self.play(Write(optimal_label), Flash(curve_dot, color=GREEN))
        
        self.wait(2)
