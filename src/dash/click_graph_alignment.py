from dash import Dash, html, dcc, Input, Output, State
import networkx as nx
import plotly.graph_objects as go
import py3Dmol as pm
import random
import MDAnalysis as mda
from MDAnalysis.analysis import align

import numpy as np
import sys
import os

def get_cluster_frame(nodes):
    frames_dict = {}
    for i in np.unique(nodes):
        frames_dict[int(i)] = np.where(nodes == i)[0]
    
    return frames_dict


def create_plotly_graph(Graph_obj, FNs):

    index_chi_node = FNs.index_chi_node
    nodes_size     = np.log(FNs.nodes_size)*50
    
    posG = nx.kamada_kawai_layout(Graph_obj, center=[0,0])

    cluster_frame = get_cluster_frame(FNs.nodes)

    edge_x, edge_y = [], []
    for edge in Graph_obj.edges():
        x0, y0 = posG[edge[0]]
        x1, y1 = posG[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y = [], []
    for node in Graph_obj.nodes():
        x, y = posG[node]
        node_x.append(x)
        node_y.append(y)
    

    node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    marker=dict(
        size=12,
        color=index_chi_node,
        colorscale='RdYlBu',
        colorbar=dict(title="Clusters"),
    ),
    hoverinfo='text',
    text=[f"Cluster {i}<br>Frame: {cluster_frame[i]}" for i in Graph_obj.nodes()], 
    customdata=[cluster_frame[i] for i in Graph_obj.nodes()],  #IMPORTANT
    )


    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )


    label_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='text',
        text=[str(i) for i in Graph_obj.nodes()],
        textposition="top center",
        hoverinfo='none',
        textfont=dict(
            size=12,
            color="black"
        )
    )

    node_trace.marker.color = index_chi_node
    node_trace.marker.size = 0.05*nodes_size



    fig = go.Figure(
    data=[edge_trace, node_trace, label_trace],
    layout=go.Layout(
        width=800,
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0)
        )   
    )

    return fig





def dash_app(Graph_obj, FNs, pdb_file, complete_traj, inp_dir, graph_name):

    plotly_graph = create_plotly_graph(Graph_obj, FNs)

    app = Dash()

    app.layout = html.Div([
        dcc.Graph(id="network-graph", figure=plotly_graph),
        html.Div(id="clicked-node-display"),
        html.Div(id="frame-dropdown-container"),

        dcc.Store(id="selected-frames", data=[]),
        dcc.Store(id="selected-cluster"),
        dcc.Store(id="cluster-frames-store"),

        dcc.Input(
            id="nth-selector",
            type="number",
            value=1,
            min=1,
            step=1,
            placeholder="Select every n-th frame"

        ),
        
        html.Button("Align Selected Frames", id="run-align", n_clicks=0),
        html.Div(id="result")
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
    [
        Output("frame-dropdown-container", "children"),
        Output("selected-cluster", "data"),
        Output("cluster-frames-store", "data")
    ],
    Input("network-graph", "clickData"),
    prevent_initial_call=True
    )
    def update_dropdown(clickData):
        if clickData is None:
            return "Click a node"

        frames = clickData["points"][0]["customdata"]
        idx = clickData["points"][0]["pointIndex"]
        cluster_node = list(Graph_obj.nodes())[idx]

        dropdown = dcc.Dropdown(
            id="frame-dropdown",
            options=[{"label": f"Frame {f}", "value": f} for f in frames],
            placeholder="Select a frame",
            multi=True,
            style={"width": "300px", "margin-top": "10px"}
        )

        return dropdown, cluster_node, frames


    
    @app.callback(
    Output("selected-frames", "data"),
    Input("frame-dropdown", "value"),
    State("selected-frames", "data"),
    prevent_initial_call=True
    )
    def add_frame_to_list(selected_frame, stored_list):

        if selected_frame is None:
            return stored_list

        if selected_frame not in stored_list:
            stored_list.append(selected_frame)

        return stored_list


    

    @app.callback(
    Output("result", "children"),
    Input("run-align", "n_clicks"),
    State("selected-frames", "data"),
    State("cluster-frames-store", "data"),
    State("selected-cluster", "data"),
    State("nth-selector", "value"),
    prevent_initial_call=True
    )
    def run_alignment(n_clicks, selected_frames, cluster_frames, cluster_node, n):

        if n_clicks == 0:
            return ""

        if selected_frames:
            frames_list= selected_frames
        else:
            frames_list= cluster_frames[::n]

        
        u = mda.Universe(
            pdb_file,
            complete_traj
        )

        ligand = u.select_atoms("resname KB8")

        for ts in u.trajectory:
            ligand.unwrap(compound="fragments")

        files = {}

        flattened_fl = []

        for l in frames_list:
            if isinstance(l, list):
                for frame in l:
                    flattened_fl.append(frame)
            else:
                flattened_fl.append(l)

        frames_list = flattened_fl


        for i in frames_list:
            fname = f"/tmp/frame_{i}.pdb"
            u.trajectory[i]
            u.atoms.write(fname)
            files[i] = fname

        ref = mda.Universe(f"/tmp/frame_{list(files.keys())[0]}.pdb")

        os.makedirs(os.path.join(inp_dir, f"{graph_name}/cluster_{cluster_node}"),  exist_ok=True)

        aligned_files = {}

        for frame, filepath in files.items():
            mob = mda.Universe(filepath)
            out = os.path.join(inp_dir, f"{graph_name}/cluster_{cluster_node}/{frame}_aligned.pdb")
            align.alignto(mob, ref, select="all")
            mob.atoms.write(out)
            aligned_files[frame] = out
        
        return f"Aligned {len(aligned_files)} frames for cluster {cluster_node}. Output: {aligned_files}"

    return app












    

