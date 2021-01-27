import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

dsgd = pd.read_csv("results/dsgd.csv")
edad = pd.read_csv("results/edad.csv")
noshare = pd.read_csv("results/noshare.csv")

full_df = pd.concat([dsgd, edad, noshare])
sb.set()
sb.lineplot(x="epoch", y="tacca", hue="mode", data=full_df)
plt.ylabel("Test AUC (Average Model)")
plt.legend(["dSGD", "edAD", "No Sharing - Each Site has 1/2 Data"])
plt.savefig("results/figures/2_sites_mnist_accuracy.png", bbox_inches="tight")
