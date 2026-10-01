from manim import *
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import *

class CTCCollapse(Scene):
    def construct(self):
        # --- SCENE 2.2: Collapse Rule ---
        title = Text("CTC Decoding: Collapse Rule", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))
        
        raw_text = Text("m  m  _  a  a  _  z  z  a", font_size=40, font="monospace")
        self.play(Write(raw_text))
        self.wait(1)
        
        # Merge repeats
        step1_title = Text("1. Merge Repeats", font_size=24, color=YELLOW).next_to(raw_text, UP, buff=0.5)
        self.play(Write(step1_title))
        
        merged_text = Text("m  _  a  _  z  a", font_size=40, font="monospace")
        self.play(TransformMatchingShapes(raw_text.copy(), merged_text))
        self.remove(raw_text)
        self.wait(1)
        
        # Drop blanks
        step2_title = Text("2. Drop Blanks (_)", font_size=24, color=YELLOW).next_to(step1_title, UP, buff=0.1)
        self.play(Write(step2_title))
        
        final_text = Text("m a z a", font_size=40, font="monospace", color=GREEN)
        self.play(TransformMatchingShapes(merged_text.copy(), final_text))
        self.remove(merged_text)
        
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])

class KenLMRescoring(Scene):
    def construct(self):
        # --- SCENE 2.5: Formula ---
        title = Text("KenLM Language Model Rescoring", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))
        
        eq = VGroup(
            Text("Score(W) = ", font_size=28),
            Text("log P_CTC(W|X) ", font_size=28),
            Text("+ α * log P_LM(W) ", font_size=28),
            Text("+ β * |W|", font_size=28)
        ).arrange(RIGHT, buff=0.1)
        self.play(Write(eq))
        
        self.play(Circumscribe(eq[1], color=BLUE, time_width=2))
        t1 = Text("Acoustic Model (Conformer)", font_size=20, color=BLUE).next_to(eq[1], DOWN)
        self.play(Write(t1))
        
        self.play(Circumscribe(eq[2], color=GOLD, time_width=2))
        t2 = Text("Language Model (KenLM)", font_size=20, color=GOLD).next_to(eq[2], DOWN).shift(RIGHT*1)
        self.play(Write(t2))
        
        self.wait(2)
        self.play(FadeOut(t1), FadeOut(t2), eq.animate.to_edge(UP).shift(DOWN*0.5))
        
        # --- SCENE 2.6: Rescoring Flip ---
        # Toy example: "तुडीच्या" vs "तुरीच्या"
        h1_label = Text("Hyp A: तुडीच्या (Acoustic error)", font_size=24, color=RED).move_to(LEFT*3 + UP*1)
        h2_label = Text("Hyp B: तुरीच्या (Valid word)", font_size=24, color=GREEN).move_to(RIGHT*3 + UP*1)
        self.play(Write(h1_label), Write(h2_label))
        
        # Acoustic scores
        a_score_1 = -4.0
        a_score_2 = -4.8
        
        # LM scores
        lm_score_1 = -12.0
        lm_score_2 = -6.0
        
        # Alpha and Beta
        alpha = 0.5
        beta = 1.0
        word_len = 3 # let's just say length penalty is +3 for both
        
        # Show CTC only first
        ctc_t1 = Text(f"CTC: {a_score_1}", font_size=24).next_to(h1_label, DOWN)
        ctc_t2 = Text(f"CTC: {a_score_2}", font_size=24).next_to(h2_label, DOWN)
        
        self.play(Write(ctc_t1), Write(ctc_t2))
        
        winner_1 = Text("Winner (Pure CTC)", font_size=20, color=RED).next_to(ctc_t1, DOWN)
        self.play(Write(winner_1))
        self.wait(1)
        
        # Add LM
        lm_t1 = Text(f"LM: {lm_score_1} * {alpha}", font_size=24, color=GOLD).next_to(ctc_t1, DOWN, buff=1.0)
        lm_t2 = Text(f"LM: {lm_score_2} * {alpha}", font_size=24, color=GOLD).next_to(ctc_t2, DOWN, buff=1.0)
        
        self.play(Write(lm_t1), Write(lm_t2))
        
        total_1 = a_score_1 + (alpha * lm_score_1) + word_len
        total_2 = a_score_2 + (alpha * lm_score_2) + word_len
        
        tot_t1 = Text(f"Total: {total_1}", font_size=28).next_to(lm_t1, DOWN)
        tot_t2 = Text(f"Total: {total_2}", font_size=28, color=GREEN).next_to(lm_t2, DOWN)
        
        self.play(Write(tot_t1), Write(tot_t2))
        
        self.play(FadeOut(winner_1))
        winner_2 = Text("New Winner (CTC + LM)", font_size=24, color=GREEN).next_to(tot_t2, DOWN)
        
        # Emphasize the flip
        self.play(Write(winner_2))
        self.play(Circumscribe(tot_t2, color=GREEN, time_width=2))
        
        self.wait(2)
