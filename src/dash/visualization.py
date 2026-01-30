import plotly.graph_objects as go
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import json
import uuid

import random
import MDAnalysis as mda
from MDAnalysis.analysis import align
import networkx as nx

import numpy as np
import sys
import os
import sys

# Add the project root
sys.path.append(os.path.abspath('../../'))
from src.dash.click_graph_alignment import*


def visualize(Graph_obj, FNs, pdb_file, complete_traj, inp_dir, traj_dir, selection):

    plotly_graph = create_plotly_graph(Graph_obj, FNs)


    os.path.join(inp_dir, pdb_file)
    os.path.join(traj_dir, complete_traj)

    u = mda.Universe(os.path.join(inp_dir, pdb_file), 
                os.path.join(traj_dir, complete_traj))

    ligand = u.select_atoms(selection)

    for ts in u.trajectory:
        ligand.unwrap(compound='fragments')


    app = Dash()

    app.layout = html.Div([
        dcc.Graph(id="network-graph", figure=plotly_graph),
        html.Div(id="clicked-node-display"),
        html.Div(id="frame-dropdown-container"),
        html.Div(id="mol3d")
    ])



    @app.callback(
    Output("clicked-node-display", "children"),
    Input("network-graph", "clickData"),
    prevent_initial_call=True
    )
    def show_clicked_node(clickData):
        if clickData is None:
            return "Click a node"
            
        idx = clickData["points"][0]["pointIndex"]
        node_id = list(Graph_obj.nodes())[idx]

        return f"Clicked node: {node_id}"



    @app.callback(
    Output("frame-dropdown-container", "children"),
    Input("network-graph", "clickData"),
    prevent_initial_call=True
    )
    def update_dropdown(clickData):
        if clickData is None:
            return "Click a node"

        # Extract frames from customdata
        frames = clickData["points"][0]["customdata"]

        return dcc.Dropdown(
            id="frame-dropdown",
            options=[{"label": f"Frame {f}", "value": f} for f in frames],
            placeholder="Select a frame",
            style={"width": "300px", "margin-top": "10px"}
        )




    @app.callback(
        Output("mol3d", "children", allow_duplicate=True),
        Input("frame-dropdown", "value"),
        prevent_initial_call=True
    )
    def show_frame(frame):

        if frame is None:
            return "Select a frame"


        fname = f"/tmp/frame_{frame}.pdb"
        u.trajectory[frame]
        u.atoms.write(fname)
        with open(fname) as f:
            pdb_data = f.read()
    
        # Render py3Dmol view
        view = pm.view(width=1000, height=800)
        view.addModel(pdb_data, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.setStyle({"resn": "KB8"}, {"stick": {}})
        
        view.zoomTo()

        # view = nv.show_structure_file(fname)

        # view.add_representation('cartoon', color='spectrum')
        # view.add_representation('ball+stick', selection='resn KB8', color='yellow', radius=1.5)
        # view.center()

        return html.Div([
            html.H4(f"Frame {frame}"),
            html.Iframe(
                srcDoc=view._make_html(),
                width="1000",
                height="800",
                style={"border": "5px solid red"},
            )
        ])

    return app


