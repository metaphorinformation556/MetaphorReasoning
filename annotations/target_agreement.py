import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv('final_with_source_pos.csv')
df_new = pd.read_csv("for_question_generation.csv")

missing_rows = df_new[~df_new['current_text'].isin(df['current_text'])]
df = df.merge(df_new[["current_text"]], on="current_text")
df = pd.concat([df, missing_rows], ignore_index=True)

cols = ['literal expression suggestion 1', 'literal expression suggestion 2', 'target']
df_clean = df

def get_metrics(col1, col2):
    s1 = df_clean[col1].astype(str)
    s2 = df_clean[col2].astype(str)
    
    percent = (s1 == s2).mean() * 100
    kappa = cohen_kappa_score(s1, s2)
    return percent, kappa

p12, k12 = get_metrics('literal expression suggestion 1', 'literal expression suggestion 2')
p1t, k1t = get_metrics('literal expression suggestion 1', 'target')
p2t, k2t = get_metrics('literal expression suggestion 2', 'target')

print(f"{'Comparison':<30} | {'Percent Agreement':<20} | {'Cohen Kappa':<12}")
print("-" * 70)
print(f"{'Annotator 1 vs Annotator 2':<30} | {p12:>17.2f}% | {k12:>11.4f}")
print(f"{'Annotator 1 vs Judge':<30} | {p1t:>17.2f}% | {k1t:>11.4f}")
print(f"{'Annotator 2 vs Judge':<30} | {p2t:>17.2f}% | {k2t:>11.4f}")
print("-" * 70)
print(f"Total rows analyzed: {len(df_clean)}")