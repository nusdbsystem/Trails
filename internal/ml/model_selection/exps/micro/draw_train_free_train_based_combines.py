from matplotlib import pyplot as plt
import warnings
import matplotlib.cbook
from matplotlib.ticker import FuncFormatter
from exps.draw_tab_lib import export_legend
import numpy as np
import matplotlib.ticker as ticker

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def draw_figure_combines(data: [], dataset: str, x_ticks: [], y_ticks: []):
    # lines' mark size
    set_marker_size = 1
    # points' mark size
    set_marker_point = 20
    # points' mark size
    set_font_size = 15
    set_lgend_size = 15
    set_tick_size = 20

    frontinsidebox = 23

    # update tick size
    matplotlib.rc('xtick', labelsize=set_tick_size)
    matplotlib.rc('ytick', labelsize=set_tick_size)

    plt.rcParams['axes.labelsize'] = set_tick_size

    mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
    mark_size_list = [set_marker_size, set_marker_size + 1, set_marker_size + 1, set_marker_size,
                      set_marker_size, set_marker_size, set_marker_size, set_marker_size + 1, set_marker_size + 2]
    line_shape_list = ['-.', '--', '-', ':']
    shade_degree = 0.2

    def thousands_formatter(x, pos):
        return '{:.3f}'.format(x)

    thousands_format = FuncFormatter(thousands_formatter)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    # draw plots
    # data is in form of time, accuracy name.,
    for i in range(len(data)):
        _annos = data[i]
        if (_annos[0] == 'JacFlow + Full Training' and dataset == "CRITEO") or _annos[0] == 'JacFlow + SUCCREJCT':
            set_marker_point = 30
        else:
            set_marker_point = 20
        if _annos[1] < 1:
            ax.plot(_annos[2] / 60, _annos[1] * 100, mark_list[i], label=_annos[0], markersize=set_marker_point)
        else:
            ax.plot(_annos[2] / 60, _annos[1], mark_list[i], label=_annos[0], markersize=set_marker_point)

    if dataset in ["C10", "C100", "IN16"]:
        plt.xscale("log")
    ax.grid()
    plt.xlabel(r"Total Time of Two Phases (min)", fontsize=set_font_size)
    if dataset in ["C10", "C100", "IN16"]:
        ax.set_ylabel(f"Test Acc on {dataset.upper()}", fontsize=set_font_size)
    else:
        ax.set_ylabel(f"Test AUC on {dataset.upper()}", fontsize=set_font_size)

    ax.xaxis.label.set_size(set_tick_size)
    ax.yaxis.label.set_size(set_tick_size)

    # ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    # tick_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # ax.set_xticks(tick_values)
    def log_formatter(x, pos):
        return f'10^{int(np.log10(x))}'

    # Set the y-axis major formatter to use your custom function
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

    # ax.set_xticklabels([f'$10^{{{int(np.log10(val))}}}$' for val in tick_values])

    if y_ticks is not None:
        if y_ticks[0] is not None:
            ax.set_ylim(bottom=y_ticks[0])
        if y_ticks[1] is not None:
            ax.set_ylim(top=y_ticks[1])
        # ax.set_ylim(y_ticks[0], y_ticks[1])
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_ticks)
    if x_ticks is not None:
        if x_ticks[0] is not None:
            ax.set_xlim(left=x_ticks[0])
        if x_ticks[1] is not None:
            ax.set_xlim(right=x_ticks[1])

    ax.yaxis.set_major_formatter(thousands_format)
    # this is for unique hash
    export_legend(fig,
                  unique_labels=["SynFlow + Full Training", "SynFlow + SUCCREJCT", "SynFlow + SUCCHALF",
                                 "SNIP + Full Training", "SNIP + SUCCREJCT", "SNIP + SUCCHALF",
                                 "JacFlow + Full Training", "JacFlow + SUCCREJCT", "JacFlow + SUCCHALF"],
                  colnum=9)

    plt.tight_layout()
    print(f"saving to combines_train_free_based_{dataset}.pdf")
    fig.savefig(f"combines_train_free_based_{dataset}.pdf", bbox_inches='tight')


dataset_res = \
    {'C100': [['JacFlow + SUCCHALF', 73.28999999999999, 14662.755894536653],
              ['JacFlow + SUCCREJCT', 72.83999999999999, 53798.99076295527],
              ['JacFlow + Full Training', 73.28999999999999, 509624.3846601936],
              ['SynFlow + SUCCHALF', 71.825, 13514.731678458074], ['SynFlow + SUCCREJCT', 72.61, 48862.90500025014],
              ['SynFlow + Full Training', 73.28999999999999, 469601.9822111476],
              ['SNIP + SUCCHALF', 69.47, 12355.488856889537], ['SNIP + SUCCREJCT', 67.695, 46188.16152468244],
              ['SNIP + Full Training', 69.94333333333333, 411390.4891362593]],
     'C10': [['JacFlow + SUCCHALF', 94.21999999999998, 14679.909915260454],
             ['JacFlow + SUCCREJCT', 94.37333333333333, 53727.87547186494],
             ['JacFlow + Full Training', 94.37333333333333, 564436.24314491],
             ['SynFlow + SUCCHALF', 94.10499999999999, 13369.244596595403],
             ['SynFlow + SUCCREJCT', 94.37333333333333, 49107.8751195788],
             ['SynFlow + Full Training', 94.37333333333333, 521005.8595802395],
             ['SNIP + SUCCHALF', 93.17999999999999, 12464.76728618767],
             ['SNIP + SUCCREJCT', 93.26666666666667, 47666.30548457928],
             ['SNIP + Full Training', 93.655, 455677.86047600623]],
     'FRAPPE': [['JacFlow + SUCCHALF', 0.9808489735757652, 3188.8936913858033],
                ['JacFlow + SUCCREJCT', 0.9814130102767649, 10475.154644335327],
                ['JacFlow + Full Training', 0.9814130102767649, 10373.3272119413],
                ['SynFlow + SUCCHALF', 0.9808489735757652, 3227.306796158371],
                ['SynFlow + SUCCREJCT', 0.9814130102767649, 10004.756667460022],
                ['SynFlow + Full Training', 0.9814130102767649, 10646.06615766388],
                ['SNIP + SUCCHALF', 0.9802515478882387, 3173.6090386758424],
                ['SNIP + SUCCREJCT', 0.9807657915745109, 9511.600655401764],
                ['SNIP + Full Training', 0.9807657915745109, 9631.877619827805]],
     'DIABETES': [['JacFlow + SUCCHALF', 0.6717315812198316, 1067.6199157399749],
                  ['JacFlow + SUCCREJCT', 0.6717315812198316, 3319.1902992887117],
                  ['JacFlow + Full Training', 0.6717315812198316, 3453.9176776570894],
                  ['SynFlow + SUCCHALF', 0.6700020728919526, 1052.334096686325],
                  ['SynFlow + SUCCREJCT', 0.6700020728919526, 3264.369818942032],
                  ['SynFlow + Full Training', 0.6700020728919526, 3411.579715983353],
                  ['SNIP + SUCCHALF', 0.6680045130878828, 1120.0702665013885],
                  ['SNIP + SUCCREJCT', 0.6680045130878828, 3584.8650441331483],
                  ['SNIP + Full Training', 0.6680045130878828, 3484.7609641713716]],
     'CRITEO': [['JacFlow + SUCCHALF', 0.8032615745641593, 60239.25669498499],
                ['JacFlow + SUCCREJCT', 0.8031991349785236, 160569.27680416164],
                ['JacFlow + Full Training', 0.8033541930059981, 138305.63400669154],
                ['SynFlow + SUCCHALF', 0.8032615745641593, 62420.96809167918],
                ['SynFlow + SUCCREJCT', 0.8031991349785236, 158649.89612360057],
                ['SynFlow + Full Training', 0.8033541930059981, 140529.48826952037],
                ['SNIP + SUCCHALF', 0.803290056967055, 58931.67735858019],
                ['SNIP + SUCCREJCT', 0.803290056967055, 157267.05170435962],
                ['SNIP + Full Training', 0.8033541930059981, 137737.27270073947]],
     'IN16': [['JacFlow + SUCCHALF', 46.35555552842882, 43857.860777873146],
               ['JacFlow + SUCCREJCT', 46.35555552842882, 160417.88986375317],
               ['JacFlow + Full Training', 46.48333334350586, 1702605.3117982924],
               ['SynFlow + SUCCHALF', 45.199999974568684, 39510.31100996213],
               ['SynFlow + SUCCREJCT', 46.40000000678168, 149907.85707761557],
               ['SynFlow + Full Training', 46.48333334350586, 1567745.5414891373],
               ['SNIP + SUCCHALF', 43.49166666666667, 37283.1842247916],
               ['SNIP + SUCCREJCT', 43.49166666666667, 141318.66606319437],
               ['SNIP + Full Training', 45.18888885498046, 1376972.6223326386]]}

for dataset, annos in dataset_res.items():
    draw_figure_combines(annos, dataset, None, None)
