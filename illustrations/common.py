from manim import *

# Palette
MALVANI_COLOR = TEAL
AHIRANI_COLOR = ORANGE
VARHADI_COLOR = PURPLE
TRUNK_COLOR = LIGHT_GREY
LM_COLOR = GOLD
TEXT_COLOR = WHITE

def make_block(label_text, color=TRUNK_COLOR, width=2.5, height=1.0):
    rect = RoundedRectangle(corner_radius=0.2, width=width, height=height, color=color, fill_opacity=0.2)
    label = Text(label_text, font_size=20, color=color)
    label.move_to(rect.get_center())
    return VGroup(rect, label)

def make_expert(name, dialect, color):
    group = VGroup()
    rect = RoundedRectangle(corner_radius=0.2, width=3.0, height=1.2, color=color, fill_opacity=0.3)
    title = Text(name, font_size=20, color=color, weight=BOLD).move_to(rect.get_center() + UP*0.2)
    sub = Text(f"({dialect})", font_size=16, color=color).next_to(title, DOWN, buff=0.1)
    group.add(rect, title, sub)
    return group

def waveform_path(width=4, height=1, num_points=100):
    import numpy as np
    x_vals = np.linspace(0, width, num_points)
    # create a somewhat random looking speech waveform envelope
    y_vals = np.sin(x_vals * 10) * np.exp(-0.5 * (x_vals - width/2)**2) * (height/2)
    # add some noise
    y_vals += np.random.normal(0, height/10, num_points)
    
    path = VMobject()
    points = [UP*y + RIGHT*x for x, y in zip(x_vals, y_vals)]
    path.set_points_smoothly(points)
    return path

def frame_strip(n=10, size=0.4, buff=0.1):
    strip = VGroup()
    for i in range(n):
        sq = Square(side_length=size, color=WHITE, fill_opacity=0)
        if i > 0:
            sq.next_to(strip[-1], RIGHT, buff=buff)
        strip.add(sq)
    return strip
