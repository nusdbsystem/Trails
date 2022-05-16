import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from exps.draw_tab_lib import export_legend


def plot_graph(lines, bars, x, filename, bar_name,
               marker_size=8, line_width=2,
               font_size=10, tick_font_size=10):
    # Configure plot settings
    plt.rcParams.update({'font.size': font_size})

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(9, 9))

    # Plot bars
    bar_width = 0.4
    hatches = ['/', '\\',  'x', 'o', 'O', '.', '*']
    colors = ['#729ECE', '#FFB579', '#98DF8A', "#FF7F7F", "#BB8FCE"]  # Softer colors #FF7F7F
    # bar_color = '#1f77b4'

    for idx, value in enumerate(bars):
        ax1.bar(x[idx], value, width=bar_width, color=colors[idx], edgecolor='black', alpha=0.5, hatch=hatches[idx])

    # Create a second y-axis on the right
    ax2 = ax1.twinx()

    # Plot lines
    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    marker_styles = ['o', 'v', 's', 'p', 'D', '*', 'h', 'H']

    for idx, line in enumerate(lines):
        ax2.plot(x, line[1], linestyle=line_styles[idx], marker=marker_styles[idx], markersize=marker_size,
                 linewidth=line_width, label=line[0])

    # Set x-axis label
    # ax1.set_xlabel("X-Axis")

    # Set y-axis labels
    ax1.set_ylabel(bar_name, fontsize=set_font_size)
    ax2.set_ylabel("SRCC Line", fontsize=set_font_size)

    # Set y-ticks
    ax1.set_yticks(np.arange(0, 16, 2))
    ax2.set_yticks(np.arange(0.3, 0.85, 0.05))

    # Set tick font size
    ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))

    ax2.set_xticklabels([8, 8, 16, 32, 64, 128], fontsize=set_font_size)
    # ax2.grid()
    ax1.grid()

    # Save the figure as a PDF
    export_legend(fig, filename="sensitive_legend", colnum=4)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


set_line_width=2
# lines' mark size
set_marker_size = 30
# points' mark size
set_marker_point = 20
# points' mark size
set_font_size = 30
set_lgend_size = 15
set_tick_size = 15

frontinsidebox = 23

# batchsize
batchsizeexps = [
    ["NTKTraceAppx", [0.32, 0.34, 0.35, 0.38, 0.37]],
    ["NTKTrace", [0.37, 0.37, 0.38, 0.38, 0.38]],
    ["Fisher", [0.38, 0.39, 0.39, 0.38, 0.39]],
    ["SNIP", [0.64, 0.64, 0.64, 0.64, 0.65]],
    ["GraSP", [0.54, 0.55, 0.53, 0.52, 0.48]],
    ["SynFlow", [0.78, 0.78, 0.78, 0.78, 0.78]],
    ["GradNorm", [0.63, 0.64, 0.64, 0.64, 0.64]],
    ["NASWOT", [0.80, 0.79, 0.79, 0.78, 0.76]]
]
bars = [0.922310, 1.845, 3.689, 7.378, 14.756]
x = [1, 2, 3, 4, 5]

plot_graph(lines=batchsizeexps, bars=bars, x=x,
           filename='./internal/ml/model_selection/exp_result/plot_batchsize.pdf',
           bar_name="GFLOPs Bar",
           marker_size=set_marker_size, line_width=set_line_width,
           font_size=set_font_size, tick_font_size=set_font_size)



# plot_channel
channelexps = [
    ["NTKTraceAppx", [0.34, 0.35, 0.36, 0.36, 0.37]],
    ["NTKTrace", [0.36, 0.38, 0.39, 0.40, 0.42]],
    ["Fisher", [0.37, 0.39, 0.40, 0.41, 0.42]],
    ["SNIP", [0.63, 0.64, 0.65, 0.65, 0.65]],
    ["GraSP", [0.45, 0.53, 0.58, 0.59, 0.59]],
    ["SynFlow", [0.77, 0.78, 0.77, 0.77, 0.76]],
    ["GradNorm", [0.63, 0.64, 0.65, 0.65, 0.65]],
    ["NASWOT", [0.80, 0.79, 0.79, 0.79, 0.78]]
]
bars = [ele/1000  for ele in [201.018, 800.746, 3273, 13076, 52288]]
x = [1, 2, 3, 4, 5]

plot_graph(lines=channelexps, bars=bars, x=x,
           filename='./internal/ml/model_selection/exp_result/plot_channel.pdf',
           bar_name = "Params Bar/Million",
           marker_size=set_marker_size, line_width=set_line_width,
           font_size=set_font_size, tick_font_size=set_font_size)
