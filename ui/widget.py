from __future__ import annotations

from typing import Any
import ipywidgets as widgets
from IPython.display import display
from indic_transformer.engine.decoder import TranslationGenerator


class TranslationWidget:
    def __init__(self, generator: TranslationGenerator, tgt_code: str) -> None:
        self.generator = generator
        self.tgt_code = tgt_code.upper()

    def render(self) -> None:
        text_box = widgets.Text(
            value="education is essential for every child",
            description="EN:",
            layout=widgets.Layout(width="500px"),
        )
        method_dropdown = widgets.Dropdown(
            options=["greedy", "beam"], value="greedy", description="Method:"
        )
        translate_btn = widgets.Button(
            description="Translate", button_style="primary"
        )
        out = widgets.Output()

        def on_click(_: widgets.Button) -> None:
            out.clear_output()
            with out:
                if method_dropdown.value == "greedy":
                    res = self.generator.greedy_decode(text_box.value)
                else:
                    res = self.generator.batched_beam_decode([text_box.value], beam_size=5)[0]
                print(f"EN : {text_box.value}")
                print(f"{self.tgt_code}: {res}")

        translate_btn.on_click(on_click)
        display(widgets.VBox([text_box, method_dropdown, translate_btn, out]))