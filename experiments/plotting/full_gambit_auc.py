import os
import glob
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

SPLIT = "class"
DATASET = "tapnet-SpokenArabicDigits"
SEARCH = "*%s*%s*" % (SPLIT, DATASET)
EXPNAME = "gru_full"
RES_FOLDER = "results"
FIG_FOLDER = "figures"
MODE = ["rankdad", "powersgd"]
MAXRANK = [3, 4, 8, 16, 32]
COLORS = dict(
    light_blue="#a6cee3",
    dark_blue="#1f78b4",
    light_green="#b2df8a",
    dark_green="#33a02c",
    light_red="#fb9a99",
    dark_red="#e31a1c",
    light_orange="#fdbf6f",
    dark_orange="#ff7f00",
    lavender="#bc80bd",
    light_purple="#cab2d6",
    dark_purple="#6a3d9a",
    yellow="#ffed6f",
    brown="#b15928",
    black="#000000",
    gray="#969696",
    pink="#fccde5",
    teal="#8dd3c7",
    salman="#fb8072",
    very_dark_orange="#b35806",
)
df = pd.DataFrame()

for csv_file in glob.iglob(os.path.join(RES_FOLDER, EXPNAME, EXPNAME + SEARCH)):
    if "k=-1" in csv_file:
        print("Skipping ", csv_file)
        continue
    print("Found ", csv_file)
    sub_df = pd.read_csv(csv_file)
    rank_index = csv_file.index("rank=")
    end_str = csv_file[(rank_index + 5) :]
    und_index = end_str.index("_")
    rank = int(end_str[:und_index])
    if rank < 3:
        continue
    site_cols = [c for c in sub_df.columns if "test_acc_site" in c]
    site_avgs = [sub_df[c] for c in site_cols]
    avg = site_avgs[0]
    if len(site_avgs) > 1:
        for s_a in site_avgs[1:]:
            avg = (avg + s_a) / 2
    sub_df["test_acc_site_avg"] = avg
    sub_df["rank"] = rank
    df = pd.concat([df, sub_df])
df = df.sort_values("mode")
# df = df[df["mode"] == MODE]
# df = df[df["rank"] == MAXRANK]
df = df[df["dataset"] == DATASET]

modes = df["mode"].unique()
ranks = df["rank"].unique()
sites = [0]
eff_cols = [c for c in df.columns if "effective_rank_site_0" in c]
rows = []
epochs = list(range(np.max(df["epoch"])))
ks = list(range(10))
EPOCH = 49
for mode in MODE:
    sub_df = df[df["mode"] == mode]
    for rank in MAXRANK:
        sub_dfr = sub_df[sub_df["rank"] == rank]
        for k in ks:
            sub_df2 = sub_dfr[sub_dfr["k"] == k]
            sub_df3 = sub_df2[sub_df2["epoch"] >= EPOCH]
            if len(sub_df3) == 0:
                continue
            row = dict(
                k=k,
                test_auc_site_avg=np.max(sub_df3["test_acc_site_avg"]),
                rankmode="%s-%s" % (mode, rank),
                mode=mode,
                rank=rank,
            )
            rows.append(row)

new_df = pd.DataFrame(rows)

sb.set()
plt.figure()
sb.boxplot(x="rankmode", y="test_auc_site_avg", data=new_df, palette="tab10")
plt.title("Max Rank=%s" % MAXRANK)
plt.ylabel("Test AUC (at last epoch)")
plt.savefig(
    "figures/full_%s_final_auc_%s.png" % (DATASET, MAXRANK), bbox_inches="tight"
)

