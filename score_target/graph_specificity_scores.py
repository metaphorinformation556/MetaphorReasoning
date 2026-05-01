import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def graph_data(d: pd.DataFrame, save_name: str) -> None:
    d = d.melt(var_name='Target Type', value_name='Specificity')
    bp = sns.boxplot(data= d, x= 'Target Type', y= 'Specificity')
    fig = bp.get_figure()
    fig.savefig("graphs/" + save_name + ".png") 
    plt.close(fig)

#data = pd.read_csv("data_with_target_specificity_scores.csv")

#print(data)

#data["Nick and Vincent Target Score 1"] = data["noun_target_score_1"]
#data["Nick and Vincent Target Score 3"] = data["noun_target_score_3"]

#data["LCC Target Lexeme Score 1"] = data["original_target_score_1"]
#data["LCC Target Lexeme Score 3"] = data["original_target_score_3"]

#score_1 = data[["Nick and Vincent Target Score 1", "LCC Target Lexeme Score 1"]]
#score_3 = data[["Nick and Vincent Target Score 3", "LCC Target Lexeme Score 3"]]
#actual_score_3 = data[["original_target_actual_score_3", "noun_target_actual_score_3"]]

#graph_data(score_1, "target_specificity_score_1_distributions")
#graph_data(score_3, "target_specificity_score_3_distributions")
#graph_data(actual_score_3, "target_normalized_actual_score_3_distributions")
