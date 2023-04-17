import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import scipy as sp
import scipy.spatial as sptl
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import distance
import pandas

def network_plot_2D(
    G,
    ax,
    plot_connections = False,
    alpha_line=0.6,
    alpha = 0.5,
    scatterpoint_size=20,
    legend=False,
    weights=False,
    edge_color="k",
    line_factor=1,
    legend_fontsize=18,
    marker_edge_color = 'k',
    marker_shape = 'o',
    marker_edge_width = 1.5,
    marker_face_color = 'r',
):

    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    # We fill each node with its attributed color. If none
    # then color the node in red.
    colors = {}
    for node in G.nodes():
        if "color" in G.nodes[node]:
            colors[node] = G.nodes[node]["color"]
        else:
            colors[node] = "tab:blue"

    if legend:

        legend = nx.get_node_attributes(G, "legend")

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    
    if plot_connections:

        for i, j in enumerate(G.edges(data=True)):

            x = np.array((pos[j[0]][1], pos[j[1]][1]))
            y = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            if weights:
                weight = j[2]["weight"] * line_factor
                plt.plot(y, x, c=edge_color, linewidth=weight, alpha=alpha_line)

            else:
                plt.plot(y, x, c=edge_color, alpha=alpha_line)

    x = []
    y = []
    nodeColor = []
    s = []
    nodelegend = []

    for key, value in pos.items():
        x.append(value[1])
        y.append(value[2])
        s.append(scatterpoint_size)
        nodeColor.append(colors[key])

        if legend:
            nodelegend.append(legend[key])

    df = pandas.DataFrame()
    df["x"] = x
    df["y"] = y
    df["s"] = s
    df["nodeColor"] = nodeColor

    if legend:
        df["legend"] = nodelegend

    groups = df.groupby("nodeColor")

    for nodeColor, group in groups:

        if legend:

            name = group.legend.unique()[0]

            ax.plot(
                group.y,
                group.x,
                marker = marker_shape,
                alpha = alpha,
                c = marker_face_color,
                markeredgewidth = marker_edge_width,
                markeredgecolor = marker_edge_color,
                linestyle="",
                ms=scatterpoint_size,
                label=name,
            )

            ax.legend(fontsize=legend_fontsize)

        else:

            ax.plot(
                group.y,
                group.x,
                marker = marker_shape,
                alpha = alpha,
                c = marker_face_color,
                markeredgewidth = marker_edge_width,
                markeredgecolor = marker_edge_color,
                linestyle="",
                ms=scatterpoint_size,
            )

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])