import os
import glob
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

SEARCH = "*class*"
EXPNAME = "rank_test_arabic"
RES_FOLDER = "results"
FIG_FOLDER = "figures"
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
if os.path.exists("experiments/full_dfs/rank_test_arabic.csv"):
    df = pd.read_csv("experiments/full_dfs/rank_test_arabic.csv")
else:
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
    df.to_csv("experiments/full_dfs/rank_test_arabic.csv", index=False)
df = df.sort_values("mode")
modes = df["mode"].unique()
ranks = df["rank"].unique()

YSTATS = dict(
    test_acc_ensemble="Test AUC (Ensemble Model)",
    test_acc_site_0="Test AUC (Site 0)",
    test_acc_avg="Test AUC (Average Model)",
    test_acc_site_avg="Test AUC (Averaged Across Sites)",
)
ALL_MODES = {
    "pooled": dict(lab="Pooled", marker="o", color=COLORS["black"]),
    "noshare": dict(
        lab="Each Site has 1/S Data - No Sharing", color=COLORS["gray"], marker="p"
    ),
    "dsgd": dict(lab="dSGD", color=COLORS["dark_blue"], marker="v"),
    "dsgd-untouched-encoder": dict(
        lab="dSGD - encoder not shared", color=COLORS["light_blue"], marker="^"
    ),  # sanity check
    "dad": dict(lab="dAD", color=COLORS["dark_red"], marker="D"),
    "dad-untouched-encoder": dict(
        lab="dAD - encoder not shared", color=COLORS["light_red"], marker="D"
    ),  # sanity check - should be equivalent ot dsgd untouched encoder
    "tedad": dict(lab="tedAD", color=COLORS["dark_orange"], marker="P"),
    "tedad-sergey": dict(
        lab="tedAD-sergey's version", color=COLORS["very_dark_orange"], marker="P"
    ),
    "tedad-all-shared": dict(
        lab="tedAD - all layers shared", color=COLORS["light_orange"], marker="P"
    ),
    "edad": dict(lab="edAD", color=COLORS["dark_purple"], marker="X"),
    "edad-untouched-encoder": dict(
        lab="edAD - encoder not shared", color=COLORS["light_purple"], marker="X"
    ),
    "rankdad": dict(lab="rank-dAD", color=COLORS["dark_green"], marker="*"),
    "rankdad-untouched-encoder": dict(
        lab="rank-dAD - encoder not shared", color=COLORS["dark_green"], marker="*"
    ),
    # "rankdad2way": dict(
    #    lab="rank-dAD-2way reduction", color=COLORS["light_green"], marker="*"
    # ),
    "powersgd": dict(lab="power-SGD", color=COLORS["brown"], marker="P"),
    "powersgd-untouched-encoder": dict(
        lab="power-SGD - encoder not shared", color=COLORS["yellow"], marker="P"
    ),
}
OMIT = []
mode_legend = {m: ALL_MODES[m] for m in modes if m in ALL_MODES and m not in OMIT}
os.makedirs(FIG_FOLDER, exist_ok=True)
for ystat, ylab in YSTATS.items():
    print("Plotting ", ystat)
    sb.set()
    fig, ax = plt.subplots(nrows=1, ncols=len(list(mode_legend.keys())), sharey=True)

    for i, (m, mode) in enumerate(mode_legend.items()):
        print("\tmode ", m)
        print("\t\tsubviding")
        sub_df = df[df["mode"] == m]
        print("\t\tplotting")
        sub_df = sub_df.sort_values("rank")
        sb.lineplot(
            x="epoch",
            y=ystat,
            data=sub_df,
            hue="rank",
            marker=mode["marker"],
            markevery=10,
            markersize=10,
            ax=ax[i],
            palette="tab10",
        )
        # ax[i].legend(["Rank-" + str(r) for r in sub_df["rank"].unique()], loc="best")
        ax[i].set_title(mode["lab"])
    ax[0].set_ylabel(ylab)

    plt.savefig(
        os.path.join(FIG_FOLDER, EXPNAME + "_%s.png" % ystat), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(FIG_FOLDER, "svg", EXPNAME + "_%s.svg" % ystat),
        bbox_inches="tight",
    )

print(modes)
