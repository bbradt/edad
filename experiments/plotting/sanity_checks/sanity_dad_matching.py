import os
import glob
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

SPLIT = "class"
SEARCH = "*%s*" % SPLIT
EXPNAME = "sanity_dad_matching"
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
)
df = pd.DataFrame()
for csv_file in glob.iglob(os.path.join(RES_FOLDER, EXPNAME, EXPNAME + SEARCH)):
    sub_df = pd.read_csv(csv_file)
    grab_cols = [sub_df[c] for c in sub_df.columns if "test_acc_site" in c]
    avg_col = grab_cols[0]
    if len(grab_cols) > 1:
        for gc in grab_cols[1:]:
            avg_col = (avg_col + gc) / 2
    sub_df["test_acc_site_avg"] = avg_col
    df = pd.concat([df, sub_df])

modes = df["mode"].unique()

YSTATS = dict(
    test_acc_ensemble="Test AUC (Ensemble)",
    test_acc_site_0="Test AUC (Site 0)",
    test_acc_avg="Test AUC (Average)",
    test_acc_site_avg="Test AUC (Averaged Over Site Performance)",
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
    "tedad": dict(lab="tedAD", color=COLORS["dark_orange"], marker="+"),
    "tedad-all-shared": dict(
        lab="tedAD - all layers shared", color=COLORS["light_orange"], marker="+"
    ),
    "edad": dict(lab="edAD", color=COLORS["dark_purple"], marker="X"),
    "edad-untouched-encoder": dict(
        lab="edAD - encoder not shared", color=COLORS["light_purple"], marker="X"
    ),
    "rankdad": dict(lab="rank-dAD", color=COLORS["dark_green"], marker="*"),
    "rankdad-untouched-encoder": dict(
        lab="rank-dAD - encoder not shared", color=COLORS["dark_green"], marker="*"
    ),
    "powersgd": dict(lab="power-SGD", color=COLORS["brown"], marker="$p$"),
    "powersgd-untouched-encoder": dict(
        lab="power-SGD - encoder not shared", color=COLORS["yellow"], marker="$p$"
    ),
}
mode_legend = {m: ALL_MODES[m] if m in ALL_MODES else m for m in modes}
os.makedirs(FIG_FOLDER, exist_ok=True)
for ystat, ylab in YSTATS.items():
    sb.set()
    plt.figure()
    for i, (m, mode) in enumerate(mode_legend.items()):
        sub_df = df[df["mode"] == m]
        sb.lineplot(
            x="epoch",
            y=ystat,
            data=sub_df,
            color=mode["color"],
            marker=mode["marker"],
            markevery=10 + i,
            markersize=10,
        )
    plt.ylabel(ylab)
    plt.legend([m["lab"] for m in mode_legend.values()])
    plt.savefig(
        os.path.join(FIG_FOLDER, EXPNAME + "_%s_%s.png" % (SPLIT, ystat)),
        bbox_inches="tight",
    )

print(modes)