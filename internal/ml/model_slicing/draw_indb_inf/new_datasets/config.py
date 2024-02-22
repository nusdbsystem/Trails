import matplotlib.pyplot as plt

bar_width = 0.35
opacity = 0.8

set_font_size = 19

set_lgend_size = 18

colors = ['#729ECE', '#8E44AD', '#2ECC71', '#ffc494', '#F39C12']
hatches = ['\\', '/', 'X', '\\\\', '//']


def export_legend(ori_fig, filename="sams_legend", colnum=9, unique_labels=[]):
    fig2 = plt.figure(figsize=(5, 0.3))
    lines_labels = [ax.get_legend_handles_labels() for ax in ori_fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # grab unique labels
    if len(unique_labels) == 0:
        unique_labels = set(labels)
    # assign labels and legends in dict
    legend_dict = dict(zip(labels, lines))
    # query dict based on unique labels
    unique_lines = [legend_dict[x] for x in unique_labels]
    fig2.legend(unique_lines, unique_labels, loc='center',
                ncol=colnum,
                fancybox=True,
                shadow=False, scatterpoints=1, fontsize=set_lgend_size)
    fig2.tight_layout()
    fig2.savefig(f"./internal/ml/model_slicing/exp_imgs/{filename}.pdf", bbox_inches='tight')
