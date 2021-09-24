import pandas as pd
import wandb
import matplotlib.pyplot as plt
import sys
from utilities import ColorPrint as uc

plot_name = ""
run_names = []
run_titles = []
smooth_factor = 0.5
reduce_rate = 1
input_csv = ""
metric = "targets_found"
limit = -1

# wandb run names are args input
args = [arg[2:] for arg in sys.argv[1:] if arg.startswith("--")]
for arg in args:
    parameter, value = arg.split("=")
    value = value.strip()
    if parameter == "run_names":
        run_names = value.split(",")
    elif parameter == "run_titles":
        run_titles = value.split(",")
    elif parameter == "smoothing":
        smooth_factor = float(value)
    elif parameter == "plot_name":
        plot_name = value
    elif parameter == "reduce_rate":
        reduce_rate = float(value)
    elif parameter == "input_csv":
        input_csv = value
    elif parameter == "metric":
        metric = value
    elif parameter == "limit":
        limit = int(value)
    else:
        continue

data = data = pd.DataFrame()

if input_csv == "": # fetching data from wandb

    uc.print_title("fetching data ...")

    # data collection
    api = wandb.Api()
    runs = api.runs("intex/ddqn")
    runs = [x for x in runs if x.name in run_names]
    for run in runs:
        history = run.scan_history()
        metric_values = [row[metric] for row in history if metric in row]
        if limit > 0:
            metric_values = metric_values[:limit]
            metric_values = metric_values
        data[run.name] = metric_values
        # data[run.name] = data[run.name] / 2.0

    # note: folder "exports_and_plots" should already exist.

    # save CSV for original data (no smoothing)
    csv_original_file = "./exports_and_plots/{}.csv".format(plot_name)
    data.to_csv(csv_original_file, sep="\t")
    uc.print_param("original CSV saved in", csv_original_file)

else:

    uc.print_title("reading data from input csv ...")

    data = pd.read_csv(input_csv)
    run_count = len(run_titles)
    ignore_colums = [0]
    for i in range(1,run_count+1):
        ignore_colums.append(i*3-1)
        ignore_colums.append(i*3)
    uc.print_param("ignoring columns", ignore_colums)
    data = data.drop(data.columns[ignore_colums], axis=1)

    data_normalized = data.loc[:, data.columns != "Step"].div(data.loc[:, data.columns != "Step"].sum(axis=1), axis=0)
    data = data_normalized


# save PNG for original data (no smoothing)
t = list(range(len(data)))

fig = plt.figure(figsize=(9, 3))
ax = fig.add_subplot()

for c in data.columns:
    ax.plot(t, data[c].to_list(), label=c, linewidth=1)

ax.set_xlabel('episode')
ax.set_ylabel(metric)
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.75))
# ax.set_title(title, fontweight="bold", fontsize="x-large", **hfont)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
fig_original_file = "./exports_and_plots/{}.png".format(plot_name)
fig.savefig(fig_original_file, bbox_inches='tight')
uc.print_param("original PNG saved in", fig_original_file)

# smoothing
for c in data.columns:
    data[c] = data[c].ewm(span=smooth_factor*100, adjust=True).mean()

# save CSV for smoothed data
csv_smoothed_file = "./exports_and_plots/{}_smoothed.csv".format(plot_name)
data.to_csv(csv_smoothed_file, sep="\t", header=run_titles)
uc.print_param("smoothed CSV saved in", csv_smoothed_file)

# save PNG for smoothed data
fig = plt.figure(figsize=(9, 3))
ax = fig.add_subplot()
for c in data.columns:
    ax.plot(t, data[c].to_list(), label=c, linewidth=1)
ax.set_xlabel('episode')
ax.set_ylabel(metric)
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.75))
# ax.set_title(title, fontweight="bold", fontsize="x-large", **hfont)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
fig_smoothed_file = "./exports_and_plots/{}_smoothed.png".format(plot_name)
fig.savefig(fig_smoothed_file, bbox_inches='tight')
uc.print_param("smoothed PNG saved in", fig_smoothed_file)

data = data[data.index % (reduce_rate * 10) == 0]

# save CSV for reduced data
csv_reduced_file = "./exports_and_plots/{}_reduced.csv".format(plot_name)
data.to_csv(csv_reduced_file, sep="\t", header=run_titles)
uc.print_param("reduced CSV saved in", csv_reduced_file)

# save PNG for reduced data
t = list(range(len(data)))

fig = plt.figure(figsize=(9, 3))
ax = fig.add_subplot()
for c in data.columns:
    ax.plot(t, data[c].to_list(), label=c, linewidth=1)
ax.set_xlabel('episode')
ax.set_ylabel(metric)
ax.legend(loc="upper left", bbox_to_anchor=(1, 0.75))
# ax.set_title(title, fontweight="bold", fontsize="x-large", **hfont)
ax.grid(axis='y', color="gainsboro")
fig.tight_layout()
fig_reduced_file = "./exports_and_plots/{}_reduced.png".format(plot_name)
fig.savefig(fig_reduced_file, bbox_inches='tight')
uc.print_param("reduced PNG saved in", fig_reduced_file)
