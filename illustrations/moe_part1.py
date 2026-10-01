from manim import *
import json
import sys
import os

# Add current dir to path to import common
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import *

class MoEArchitecture(Scene):
    def construct(self):
        # --- SCENE 1.2: Block Anatomy ---
        title = Text("Dialect-Aware MoE Conformer Block", font_size=36, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Build Trunk (Left)
        trunk_group = VGroup()
        hidden = make_block("Hidden States (x)", width=3, height=0.8).move_to(LEFT*4 + UP*1.5)
        ln = make_block("LayerNorm", width=3, height=0.8).next_to(hidden, DOWN, buff=0.5)
        
        a1 = Arrow(hidden.get_bottom(), ln.get_top(), buff=0.1)
        
        self.play(Create(hidden))
        self.play(GrowArrow(a1), Create(ln))
        
        # Split into Shared FFN and Router
        shared_ffn = make_block("Shared FFN\n(Global Patterns)", color=TRUNK_COLOR, width=3, height=1.2)
        shared_ffn.next_to(ln, DOWN, buff=1.5)
        
        router = make_block("Routing Network", color=YELLOW, width=3, height=0.8)
        router.move_to(RIGHT*2 + UP*0.5)
        
        a2 = Arrow(ln.get_bottom(), shared_ffn.get_top(), buff=0.1)
        
        # Path to router bends right
        router_path = CurvedArrow(ln.get_right(), router.get_left(), angle=-TAU/8, color=YELLOW)
        
        self.play(GrowArrow(a2), Create(shared_ffn))
        self.play(Create(router_path), Create(router))
        
        # Build Experts (Right)
        experts = VGroup()
        exp1 = make_expert("Expert 1", "Malvani", MALVANI_COLOR)
        exp2 = make_expert("Expert 2", "Ahirani", AHIRANI_COLOR)
        exp3 = make_expert("Expert 3", "Varhadi", VARHADI_COLOR)
        
        experts.add(exp1, exp2, exp3).arrange(DOWN, buff=0.4)
        experts.next_to(router, DOWN, buff=1.0).shift(RIGHT*1)
        
        self.play(LaggedStart(*[Create(e) for e in experts], lag_ratio=0.3))
        
        # Arrows from router to experts
        r_arrows = VGroup()
        for e in experts:
            r_arrows.add(Arrow(router.get_bottom(), e.get_left(), buff=0.1, color=YELLOW))
            
        self.play(LaggedStart(*[GrowArrow(a) for a in r_arrows], lag_ratio=0.1))
        
        self.wait(1)
        
        # --- SCENE 1.3 & 1.4: Router gating & Activation (Soft-Gating Example) ---
        # Let's show soft-gating on a mixed utterance
        # Probabilities: 0.40, 0.35, 0.25
        gating_title = Text("Soft-Gating (Weighted Activation)", font_size=24, color=YELLOW).next_to(router, UP)
        self.play(Write(gating_title))
        
        probs = [0.40, 0.35, 0.25]
        labels = VGroup()
        for i, (e, p) in enumerate(zip(experts, probs)):
            lbl = Text(f"{p:.2f}", font_size=20, color=YELLOW).next_to(r_arrows[i], UP, buff=0.1)
            labels.add(lbl)
            
        self.play(Write(labels))
        
        # Dim experts according to weights (Soft-gating)
        self.play(
            exp1.animate.set_opacity(probs[0] + 0.3),
            exp2.animate.set_opacity(probs[1] + 0.3),
            exp3.animate.set_opacity(probs[2] + 0.3),
        )
        
        self.play(Indicate(exp1, color=MALVANI_COLOR), Indicate(exp2, color=AHIRANI_COLOR))
        self.wait(1)
        
        # --- SCENE 1.5: Fusion ---
        fusion_node = Circle(radius=0.4, color=WHITE).move_to(DOWN*2.5 + LEFT*1)
        plus = Text("+", font_size=36).move_to(fusion_node.get_center())
        fusion = VGroup(fusion_node, plus)
        
        self.play(Create(fusion))
        
        # From Shared FFN
        f_a1 = Arrow(shared_ffn.get_right(), fusion_node.get_left(), buff=0.1)
        # From Experts (we'll just draw one big arrow representing the weighted sum, or arrows from all 3)
        f_a2 = Arrow(experts.get_bottom(), fusion_node.get_right(), buff=0.1)
        
        self.play(GrowArrow(f_a1), GrowArrow(f_a2))
        
        eq = Text("y = F_shared(x) + Sum(g_k * F_k(x))", font_size=24)
        eq.next_to(fusion, DOWN, buff=0.5)
        self.play(Write(eq))
        
        self.wait(2)
        
        # Clean up for next scene
        self.play(
            *[FadeOut(m) for m in self.mobjects]
        )
        self.wait(1)

class MoEPerFrame(Scene):
    def construct(self):
        # --- SCENE 1.6: Per-frame routing ---
        title = Text("Per-Frame Dialect Routing", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))
        
        # Load real data
        try:
            with open("illustrations/moe_activations.json", "r") as f:
                data = json.load(f)
        except Exception:
            # Fallback mock data if json missing
            data = {"D1": {"probs": [[0.8,0.1,0.1], [0.7,0.2,0.1], [0.9,0.05,0.05]]}}
            
        d_key = "D1" # Malvani utterance
        if d_key in data:
            utterance_probs = data[d_key]["probs"]
        else:
            utterance_probs = [[0.8,0.1,0.1], [0.7,0.2,0.1], [0.9,0.05,0.05]]
            
        # We will show 20 frames as a strip
        n_frames = min(20, len(utterance_probs))
        strip = frame_strip(n=n_frames, size=0.5).move_to(ORIGIN)
        
        self.play(Create(strip))
        
        # Colors: 0=Malvani, 1=Ahirani, 2=Varhadi
        expert_colors = [MALVANI_COLOR, AHIRANI_COLOR, VARHADI_COLOR]
        
        legend = VGroup(
            Text("Malvani Expert", font_size=20, color=MALVANI_COLOR),
            Text("Ahirani Expert", font_size=20, color=AHIRANI_COLOR),
            Text("Varhadi Expert", font_size=20, color=VARHADI_COLOR)
        ).arrange(RIGHT, buff=1).next_to(title, DOWN, buff=1)
        self.play(Write(legend))
        
        self.wait(1)
        
        # Color each frame based on argmax (Top-1 representation)
        animations = []
        for i in range(n_frames):
            probs = utterance_probs[i]
            best_idx = probs.index(max(probs))
            animations.append(strip[i].animate.set_fill(expert_colors[best_idx], opacity=0.8))
            
        self.play(LaggedStart(*animations, lag_ratio=0.1, run_time=2))
        
        self.wait(2)
