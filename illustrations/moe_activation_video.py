import json
from manim import *

class MoEActivation(Scene):
    def construct(self):
        with open("illustrations/moe_activations.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        title = Text("MoE Expert Routing Activation", font_size=40)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        expert_names = ["Malvani (D1)", "Ahirani (D2)", "Varhadi (D4)"]
        colors = [BLUE, GREEN, RED]
        
        for d_key, d_name in zip(["D1", "D2", "D4"], ["Malvani (D1)", "Ahirani (D2)", "Varhadi (D4)"]):
            if d_key not in data:
                continue
            dialect_data = data[d_key]
            
            d_label = Text(f"Input Audio Dialect: {d_name}", font_size=32).next_to(title, DOWN)
            self.play(Write(d_label))
            
            txt = dialect_data["text"]
            if len(txt) > 50: txt = txt[:47] + "..."
            t_label = Text(txt, font_size=24, color=YELLOW).next_to(d_label, DOWN)
            self.play(Write(t_label))
            
            probs = dialect_data["probs"]
            
            def create_bars(vals):
                group = VGroup()
                for i, (val, color, name) in enumerate(zip(vals, colors, expert_names)):
                    # val is between 0 and 1
                    h = max(val * 4, 0.01) # max height 4
                    bar = Rectangle(width=1.5, height=h, color=color, fill_opacity=0.8)
                    bar.move_to(DOWN*1.5 + RIGHT*(i-1)*2.5)
                    bar.align_to(DOWN*3.5, DOWN) # bottom align
                    
                    label = Text(name, font_size=20).next_to(bar, DOWN)
                    label.align_to(DOWN*3.8, DOWN)
                    
                    percent = Text(f"{val:.2f}", font_size=20).next_to(bar, UP)
                    
                    group.add(VGroup(bar, label, percent))
                return group

            chart = create_bars(probs[0])
            self.play(Create(chart))
            
            # Animate over frames
            for i in range(1, len(probs)):
                new_chart = create_bars(probs[i])
                self.play(Transform(chart, new_chart), run_time=0.1)
                
            self.wait(1)
            self.play(FadeOut(chart), FadeOut(d_label), FadeOut(t_label))
            
        self.wait(1)
