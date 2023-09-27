from manim import *

class DynamicBatching(Scene):
    def construct(self):
        # Text labels
        title = Text("Dynamic Batching in TorchServe").scale(0.75).to_edge(UP)
        batch_size_text = Text("Max Batch Size Set to 4").scale(0.6).next_to(title, DOWN).set_color(YELLOW)
        batch_delay_text = Text("Max Batch Delay Exceeded").scale(0.6).next_to(batch_size_text, DOWN).set_color(BLUE)
        
        # Adjust the queue positions to raise the initial render requests
        queue_positions = [UP + DOWN * 1.5 * i for i in range(5)]
        requests = [Rectangle(height=1, width=2.5, fill_opacity=0.5, fill_color=BLUE) for _ in range(3)]
        requests += [Rectangle(height=1, width=2.5, fill_opacity=0.5, fill_color=YELLOW) for _ in range(2)]
        request_labels = [Text(f"Request {i+1}").scale(0.5).move_to(req) for i, req in enumerate(requests)]
        for request, label, position in zip(requests, request_labels, queue_positions):
            request.move_to(position)
            label.move_to(position)
        
        # Batch box
        batch_box = Rectangle(height=5.5, width=3).to_edge(LEFT)
        batch_text = Text("Batch").scale(0.6).next_to(batch_box, UP)
        
        # Move the Serve box further to the right
        serve_box = Rectangle(height=4, width=3).next_to(batch_box, RIGHT, buff=7)
        serve_text = Text("Run inference").scale(0.6).move_to(serve_box)
        
        arrow = Arrow(batch_box.get_right(), serve_box.get_left(), buff=0.2)
        
        self.play(Write(title))
        self.wait(0.5)
        
        # Animate requests coming into the queue
        for i, (rect, label) in enumerate(zip(requests, request_labels)):
            self.play(FadeIn(rect), Write(label))
            self.wait(0.5)
            if i == 2:  # Max batch size exceeded for demo
                self.play(Write(batch_size_text))
                batch = VGroup(*requests[:3], *request_labels[:3])
                self.play(batch.animate.move_to(batch_box), FadeIn(batch_box), Write(batch_text))
                self.wait(1.5)
        
        # Show the arrow and the "Max Batch Delay Exceeded" text together
        self.play(Create(arrow), Write(batch_delay_text))
        
        self.wait(1.5)
        
        self.play(FadeIn(serve_box), Write(serve_text))
        
        self.wait(1.5)
        
        self.play(FadeOut(VGroup(*requests, serve_box, serve_text, title, batch_size_text, batch_delay_text, *request_labels, batch_box, batch_text, arrow)))

# Render the scene
scene = DynamicBatching()
scene.render()
