"""Plotly styling that matches the app theme."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from ui import theme

COLORWAY = [theme.GREEN, theme.LILAC, theme.ORANGE, theme.PINK, theme.SKY, "#f5e663", theme.RISK]
px.defaults.color_discrete_sequence = COLORWAY   # plotly express picks colours at creation time


def show(fig, height: int | None = None) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, system-ui, sans-serif", color=theme.CREAM, size=13),
        colorway=COLORWAY, margin=dict(l=64, r=16, t=40, b=56),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor=theme.SURFACE, bordercolor=theme.LINE, font_color=theme.CREAM),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    if height:
        fig.update_layout(height=height)
    grid = "rgba(255,252,225,0.07)"
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=theme.LINE, color=theme.MUTED, automargin=True,
                     title_font_color=theme.MUTED)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=theme.LINE, color=theme.MUTED, automargin=True,
                     title_font_color=theme.MUTED)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})
