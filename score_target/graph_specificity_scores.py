import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def graph_data(d: pd.DataFrame, save_name: str) -> None:
    d = d.melt(var_name='Target Type', value_name='Specificity')
    bp = sns.boxplot(data= d, x= 'Target Type', y= 'Specificity')
    fig = bp.get_figure()
    fig.savefig("graphs/" + save_name + ".png") 
    plt.close(fig)
