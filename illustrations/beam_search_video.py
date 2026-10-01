import json
from manim import *

class BeamSearchIllustration(Scene):
    def construct(self):
        title = Text("Greedy vs. Beam Search", font_size=40)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        greedy_title = Text("Greedy Decoding", font_size=32).next_to(title, DOWN)
        self.play(Write(greedy_title))
        
        root = Circle(radius=0.3, color=BLUE).shift(LEFT*3 + UP*1)
        root_txt = Text("t=0", font_size=20).move_to(root)
        self.play(Create(root), Write(root_txt))
        
        n1_1 = Circle(radius=0.4, color=GREEN).shift(LEFT*1 + UP*2)
        n1_2 = Circle(radius=0.4, color=RED).shift(LEFT*1 + UP*0)
        t1_1 = Text("र (0.4)", font_size=20).move_to(n1_1)
        t1_2 = Text("ड (0.6)", font_size=20).move_to(n1_2)
        
        e1_1 = Line(root.get_right(), n1_1.get_left())
        e1_2 = Line(root.get_right(), n1_2.get_left())
        
        self.play(Create(e1_1), Create(e1_2), Create(n1_1), Create(n1_2), Write(t1_1), Write(t1_2))
        
        hl1 = n1_2.copy().set_color(YELLOW).set_stroke(width=4)
        self.play(Create(hl1))
        
        n2_1 = Circle(radius=0.4, color=RED).shift(RIGHT*1 + UP*1)
        n2_2 = Circle(radius=0.4, color=GREEN).shift(RIGHT*1 + DOWN*1)
        t2_1 = Text("ा (0.3)", font_size=20).move_to(n2_1)
        t2_2 = Text("ी (0.7)", font_size=20).move_to(n2_2)
        
        e2_1 = Line(n1_2.get_right(), n2_1.get_left())
        e2_2 = Line(n1_2.get_right(), n2_2.get_left())
        
        self.play(Create(e2_1), Create(e2_2), Create(n2_1), Create(n2_2), Write(t2_1), Write(t2_2))
        
        hl2 = n2_2.copy().set_color(YELLOW).set_stroke(width=4)
        self.play(Create(hl2))
        
        g_prob = Text("Path Prob (तुडी): 0.6 x 0.7 = 0.42", font_size=24, color=YELLOW).shift(RIGHT*3 + DOWN*2)
        self.play(Write(g_prob))
        
        self.wait(2)
        self.play(FadeOut(greedy_title), FadeOut(hl1), FadeOut(hl2), FadeOut(g_prob))
        
        beam_title = Text("Beam Search (Beam Width = 2)", font_size=32).next_to(title, DOWN)
        self.play(Write(beam_title))
        
        bh1 = n1_1.copy().set_color(YELLOW).set_stroke(width=4)
        bh2 = n1_2.copy().set_color(YELLOW).set_stroke(width=4)
        self.play(Create(bh1), Create(bh2))
        
        n2_3 = Circle(radius=0.4, color=GREEN).shift(RIGHT*1 + UP*3)
        n2_4 = Circle(radius=0.4, color=RED).shift(RIGHT*1 + UP*2)
        t2_3 = Text("ी (0.9)", font_size=20).move_to(n2_3)
        t2_4 = Text("ु (0.1)", font_size=20).move_to(n2_4)
        
        e2_3 = Line(n1_1.get_right(), n2_3.get_left())
        e2_4 = Line(n1_1.get_right(), n2_4.get_left())
        
        self.play(Create(e2_3), Create(e2_4), Create(n2_3), Create(n2_4), Write(t2_3), Write(t2_4))
        
        p1 = Text("तुरी: 0.4x0.9=0.36", font_size=20).next_to(n2_3, RIGHT)
        p2 = Text("तुरु: 0.4x0.1=0.04", font_size=20).next_to(n2_4, RIGHT)
        p3 = Text("तुडा: 0.6x0.3=0.18", font_size=20).next_to(n2_1, RIGHT)
        p4 = Text("तुडी: 0.6x0.7=0.42", font_size=20).next_to(n2_2, RIGHT)
        
        self.play(Write(p1), Write(p2), Write(p3), Write(p4))
        
        bhl1 = n2_2.copy().set_color(YELLOW).set_stroke(width=4)
        bhl2 = n2_3.copy().set_color(YELLOW).set_stroke(width=4)
        self.play(Create(bhl1), Create(bhl2))
        
        final_winner = Text("Top 2 Hypotheses: तुडी (0.42) & तुरी (0.36)", font_size=24, color=GREEN).shift(RIGHT*3 + DOWN*2)
        self.play(Write(final_winner))
        
        self.wait(2)
