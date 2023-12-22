import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from nicegui import ui

state = {}


def update_plot():
    with state["plot"]:
        plt.cla()
        if "counter" not in state:
            state["counter"] = 5.0
        state["counter"] += 1
        actual_plot(x_length=state["counter"])


def main():
    ui.label("This is a factor graph.")
    ui.button("Click me!", on_click=update_plot)

    with ui.row() as row:
        state["row"] = row
        draw_pyplot()
        draw_updating_line_plot()

    ui.run()


def draw_updating_line_plot():
    line_plot = ui.line_plot(n=2, limit=20, figsize=(3, 2), update_every=5).with_legend(
        ["sin", "cos"], loc="upper center", ncol=2
    )

    def update_line_plot() -> None:
        now = datetime.now()
        x = now.timestamp()
        y1 = math.sin(x)
        y2 = math.cos(x)
        line_plot.push([now], [[y1], [y2]])

    line_updates = ui.timer(0.1, update_line_plot, active=False)
    line_checkbox = ui.checkbox("active").bind_value(line_updates, "active")


def draw_pyplot():
    with ui.pyplot(figsize=(15, 10)) as plot:
        state["plot"] = plot
        actual_plot()


def actual_plot(x_length=5.0):
    x = np.linspace(0.0, x_length)
    y = np.cos(2 * np.pi * x) * np.exp(-x)
    plt.plot(x, y, "-")


if __name__ in {"__main__", "__mp_main__"}:
    main()
