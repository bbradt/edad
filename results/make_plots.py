import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

full_df = pd.read_csv("results/all_methods.csv")

dsgd_df = full_df[full_df["mode"] == "dsgd"]
edad_df = full_df[full_df["mode"] == "edad"]
noshare_df = full_df[full_df["mode"] == "noshare"]
pooled_df = full_df[full_df["mode"] == "pooled"]

sb.set()
sb.lineplot(x="epoch", y="tacca", data=dsgd_df, marker="o", markevery=8, markersize=10)
sb.lineplot(x="epoch", y="tacca", data=edad_df, marker="v", markevery=9, markersize=10)
sb.lineplot(
    x="epoch", y="tacca", data=noshare_df, marker="s", markevery=10, markersize=10
)
sb.lineplot(
    x="epoch", y="tacca", data=pooled_df, marker="X", markevery=11, markersize=10
)
plt.ylabel("Test AUC (Average Model)")
plt.legend(["dSGD", "edAD", "No Sharing - Each Site has 1/2 Data", "Pooled"])
plt.savefig("results/figures/2_sites_class_mnist_accuracy.png", bbox_inches="tight")
