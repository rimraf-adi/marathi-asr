import json
from manim import *

class KenLMIllustration(Scene):
    def construct(self):
        title = Text("Decoding: CTC Only vs. CTC + KenLM", font_size=40)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))

        ctc_title = Text("CTC Beam Search (Acoustic Only)", font_size=32).next_to(title, DOWN)
        self.play(Write(ctc_title))
        
        h1 = Text("Hypothesis 1 (Greedy): 'लवकर तयार होणाऱ्या तुडीच्या जाती कोणत्या ?'\n(tur vs tudi error)\nCTC Score: 0.8", font_size=20).shift(UP*1)
        h2 = Text("Hypothesis 2 (Correct): 'लवकर तयार होणाऱ्या तुरीच्या जाती कोणत्या ?'\n(Valid Marathi Crop Word)\nCTC Score: 0.6", font_size=20).next_to(h1, DOWN, buff=1)
        
        self.play(Write(h1), Write(h2))
        
        winner1 = Text("Winner without LM: Hypothesis 1 (Acoustic error wins)", color=RED, font_size=24).next_to(h2, DOWN, buff=1)
        self.play(Write(winner1))
        self.wait(2)
        
        self.play(FadeOut(ctc_title), FadeOut(winner1))
        
        lm_title = Text("Adding KenLM (Language Model)", font_size=32).next_to(title, DOWN)
        self.play(Write(lm_title))
        
        lm1 = Text("LM Score: 0.1", color=YELLOW, font_size=24).next_to(h1, DOWN, buff=0.2)
        lm2 = Text("LM Score: 0.9", color=YELLOW, font_size=24).next_to(h2, DOWN, buff=0.2)
        
        self.play(Write(lm1), Write(lm2))
        
        t1 = Text("Total = 0.8 + 0.1 = 0.9", color=GREEN, font_size=24).next_to(lm1, RIGHT, buff=0.5)
        t2 = Text("Total = 0.6 + 0.9 = 1.5", color=GREEN, font_size=24).next_to(lm2, RIGHT, buff=0.5)
        
        self.play(Write(t1), Write(t2))
        
        winner2 = Text("Winner: Hypothesis 2 (KenLM rescues the valid word)", color=GREEN, font_size=24).next_to(h2, DOWN, buff=1)
        self.play(Write(winner2))
        
        self.wait(2)
