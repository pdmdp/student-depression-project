"""Render the two README figures from the recorded review metrics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
data=json.loads((root/"results/review.json").read_text())
names=list(data["metrics"])
labels=["Majority baseline", "L1 logistic", "Without suicidal-thought item"]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11})
fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout="constrained",sharey=True)
for ax,metric in zip(axes,["F1 (label 1)","Balanced accuracy"]):
    bars=ax.barh(labels,[data["metrics"][n][metric] for n in names],color=["#a5b6c5","#156b58","#317c9c"],height=.55)
    ax.bar_label(bars,fmt="%.3f",padding=5)
    ax.set(xlim=(0,1.03),title=metric)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
axes[0].invert_yaxis()
(root/"images").mkdir(exist_ok=True)
fig.savefig(root/"images/review-comparison.png",dpi=180,facecolor="white")
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),layout="constrained")
for ax,name,title in zip(axes,names[1:],labels[1:]):
    matrix=np.array(data["matrices"][name])
    ax.imshow(matrix,cmap="Blues",vmin=0,vmax=3300)
    for (y,x),v in np.ndenumerate(matrix):ax.text(x,y,f"{v:,}",ha="center",va="center",fontsize=15,color="white" if v>1700 else "#192f3e")
    ax.set(xticks=[0,1],yticks=[0,1],xlabel="Predicted label",ylabel="True label",title=title)
fig.savefig(root/"images/review-confusion-matrices.png",dpi=180,facecolor="white")
