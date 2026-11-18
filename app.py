"""Gradio demo: draw or upload a garment sketch, get design options.

    python app.py            # after training (checkpoints/ethnicwear_bicycle_gan/)
"""

import gradio as gr

from sketch2design.generator import Designer

designer = Designer()


def explore(sketch, n, seed):
    if sketch is None:
        raise gr.Error("Upload a sketch first.")
    return designer.random_designs(sketch, n=int(n), seed=int(seed))


def match(sketch, reference):
    if sketch is None or reference is None:
        raise gr.Error("Upload both a sketch and a reference photo.")
    return designer.match_reference(sketch, reference)


with gr.Blocks(title="Sketch to ethnic-wear design") as demo:
    gr.Markdown(
        "## Sketch → ethnic-wear design\n"
        "Upload a line sketch of a saree, kurta or lehenga (a phone photo of paper works). "
        "**Explore** gives several colourways for the same silhouette. **Match a reference** "
        "borrows the colours and fabric of a photo you like."
    )
    with gr.Row():
        sketch = gr.Image(type="pil", label="Your sketch", image_mode="RGB")
        with gr.Column():
            n = gr.Slider(2, 12, value=6, step=1, label="How many options")
            seed = gr.Number(value=0, precision=0, label="Seed (change for new options)")
            go = gr.Button("Explore designs", variant="primary")
    gallery = gr.Gallery(label="Design options", columns=6, height="auto")
    go.click(explore, [sketch, n, seed], gallery)

    gr.Markdown("### Match a reference")
    with gr.Row():
        reference = gr.Image(type="pil", label="Reference garment or fabric photo")
        matched = gr.Image(type="pil", label="Your sketch in that style")
    gr.Button("Match style").click(match, [sketch, reference], matched)

if __name__ == "__main__":
    demo.launch(share=True)
