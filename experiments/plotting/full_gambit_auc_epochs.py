import os
import glob
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

SPLIT = "class"
DATASETS = [
    "tapnet-PenDigits",
    "tapnet-CharacterTrajectories",
    "tapnet-SpokenArabicDigits",
    "tapnet-Epilepsy",
    "tapnet-NATOPS",
    "tapnet-PEMS-SF",
    "tapnet-Heartbeat",
    "mnist",
]
for DATASET in DATASETS:
    # DATASET = "tapnet-SpokenArabicDigits"
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
    sb.set()
    fig, ax = plt.subplots(1, len(MODE), sharey=True)
    for i, mode in enumerate(MODE):
        axi = ax[i]
        sub_df = df[df["mode"] == mode]
        sb.lineplot(
            x="epoch",
            y="test_acc_site_avg",
            hue="rank",
            data=sub_df,
            ax=axi,
            palette="tab10",
        )
        axi.set_title(mode)
        axi.set_xlabel("")
        axi.set_ylabel("")
    # plt.legend(None)
    plt.savefig("figures/full_%s_final_auc_epochs.png" % (DATASET), bbox_inches="tight")
    plt.savefig(
        "figures/svg/full_%s_final_auc_epochs.svg" % (DATASET), bbox_inches="tight"
    )

