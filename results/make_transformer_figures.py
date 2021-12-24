import os
import glob
from numpy.core.numeric import full
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

modes = ["pooled", "dad", "dsgd", "rankdad"]
batches = [16]
ks = [0, 1, 2, 3, 4]

form = "transformer_model=transformer_mode=%s_sites=2_split=random_data=imdb_lr=0.0001_batch_size=%s_seed=0_k=%s_kf=5_rank=8_numiterations=10.csv"

results = "results/transformer"

full_df = pd.DataFrame()

for mode in modes:
    for batch in batches:
        if mode == "pooled":
            batch *= 2
        for k in ks:
            filename = os.path.join(results, form % (mode, batch, k))
            df = pd.read_csv(filename)
            full_df = pd.concat([full_df, df]).reset_index(drop=True)
site_avg = None
for site in range(1):
    site_df = full_df["test_acc_site_%d" % site]
    if site_avg is None:
        site_avg = site_df
    else:
        site_avg += site_df
        site_avg /= 2
full_df["Average AUC"] = site_avg
full_df = full_df[full_df["epoch"] < 50]

plt.figure()
sb.set()
sb.lineplot(x="epoch", y="Average AUC", hue="mode", data=full_df)
plt.ylim(0, 1)
plt.savefig("figures/transformer_final.png", bbox_inches="tight")
