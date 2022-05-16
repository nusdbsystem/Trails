import matplotlib.pyplot as plt
import json
from matplotlib.ticker import FuncFormatter

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 20  # Set the font size
set_lgend_size = 15
set_tick_size = 20
cpu_colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']
gpu_colors = ['#98DF8A', '#D62728', '#1ABC9C', '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '//', '\\\\',
           '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

base_dir = "./internal/ml/model_selection/exp_current_filter_cache/"

workers = [1, 2, 3, 4, 8]
latencies = []

for worker in workers:
    file_name = f"{base_dir}/time_score_mlp_sp_frappe_batch_size_32_cpu_{worker}.json"
    with open(file_name, 'r') as file:
        latency = json.load(file)[0]  # Assuming it's a list with one value
        latencies.append(latency)

fig, ax = plt.subplots()
bars = plt.bar(workers, latencies, bar_width, alpha=opacity, color=cpu_colors[:len(workers)])

plt.xlabel('Number of Workers', fontsize=set_font_size)
plt.ylabel('Latency (s)', fontsize=set_font_size)
plt.title('Latency by Number of Workers', fontsize=set_font_size)
plt.xticks(workers, fontsize=set_tick_size)
plt.yticks(fontsize=set_tick_size)
plt.legend(fontsize=set_lgend_size)
plt.tight_layout()
plt.show()

fig.savefig(f"./internal/ml/model_selection/exp_result/concurrent_latency.pdf", bbox_inches='tight')

# plot memory usage here
memory_usages = []
peak_memory_usages = []  # Store peak memory usages here

for worker in workers:
    file_name = f"{base_dir}/resource_score_mlp_sp_frappe_batch_size_32_cpu_{worker}.json"
    with open(file_name, 'r') as file:
        memory_usage = json.load(file)['memory_usage']
        memory_usages.append(memory_usage)
        peak_memory_usages.append(max(memory_usage))  # Compute peak memory for this worker

fig, ax = plt.subplots()
box = plt.boxplot(memory_usages, patch_artist=True, showfliers=True)

for patch, color, hatch in zip(box['boxes'], cpu_colors[:len(workers)], hatches[:len(workers)]):
    patch.set_facecolor(color)
    patch.set_hatch(hatch)

plt.plot(range(1, len(workers) + 1), peak_memory_usages, marker='o', linestyle='-', color='black',
         label='Peak Memory Usage')  # Plot the peak memory line

plt.xlabel('Number of Workers', fontsize=set_font_size)


def thousands_formatter(x, pos):
    return f'{int(x / 1000)}k'


formatter = FuncFormatter(thousands_formatter)
plt.ylabel('Memory Usage (k)', fontsize=set_font_size)
plt.gca().yaxis.set_major_formatter(formatter)

plt.title('Memory Usage by Number of Workers', fontsize=set_font_size)
plt.xticks(ticks=range(1, len(workers) + 1), labels=workers, fontsize=set_tick_size)
plt.yticks(fontsize=set_tick_size)
plt.legend(loc='best', fontsize=set_lgend_size)  # Include the new line in the legend
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig(f"./internal/ml/model_selection/exp_result/concurrent_memory.pdf", bbox_inches='tight')
