from scipy.stats import mannwhitneyu, ttest_rel, spearmanr, pearsonr, ks_2samp
import pandas as pd

def get_mannwhitney_p_value(dist_1: list, dist_2: list) -> float:
    test = mannwhitneyu(dist_1, dist_2, alternative= "two-sided")
    return test.pvalue

def get_t_p_value(dist_1: list, dist_2: list) -> float:
    test = ttest_rel(dist_1, dist_2)
    return test.pvalue

def get_spearman_corr(dist_1: list, dist_2: list) -> tuple:
    corr, p_value = spearmanr(dist_1, dist_2)
    return corr, p_value

def get_pearson_corr(dist_1: list, dist_2: list) -> tuple:
    corr, p_value = pearsonr(dist_1, dist_2)
    return corr, p_value

def get_ks_p_value(dist_1: list, dist_2: list) -> float:
    test = ks_2samp(dist_1, dist_2)
    return test.pvalue

data = pd.read_csv("data_with_target_specificity_scores.csv")

score_1_original = data["original_target_score_1"].tolist()
score_1_noun = data["noun_target_score_1"].tolist()

score_3_original = data["original_target_score_3"].tolist()
score_3_noun = data["noun_target_score_3"].tolist()

actual_score_3_original = data["original_target_actual_score_3"].tolist()
actual_score_3_noun = data["noun_target_actual_score_3"].tolist()

# Score 1
mw_1 = get_mannwhitney_p_value(score_1_original, score_1_noun)
t_1 = get_t_p_value(score_1_original, score_1_noun)
r_1, r_1_p = get_spearman_corr(score_1_original, score_1_noun)

# Score 3
mw_3 = get_mannwhitney_p_value(score_3_original, score_3_noun)
t_3 = get_t_p_value(score_3_original, score_3_noun)
r_3, r_3_p = get_spearman_corr(score_3_original, score_3_noun)

print(
    f"Score 1 - Mann–Whitney P-Value: {mw_1}, "
    f"Paired t-test P-Value: {t_1}, "
    f"Spearman r: {r_1}, Spearman P-Value: {r_1_p}"
)

print(
    f"Score 3 - Mann–Whitney P-Value: {mw_3}, "
    f"Paired t-test P-Value: {t_3}, "
    f"Spearman r: {r_3}, Spearman P-Value: {r_3_p}"
)

r_orig, p_orig = get_spearman_corr(
    score_1_original,
    score_3_original
)

r_noun, p_noun = get_spearman_corr(
    score_1_noun,
    score_3_noun
)

print(
    f"Original condition - Interscore Spearman r: {r_orig}, "
    f"P-Value: {p_orig}"
)

print(
    f"Noun condition - Interscore Spearman r: {r_noun}, "
    f"P-Value: {p_noun}"
)

r_orig_p, p_orig_p = pearsonr(
    score_1_original,
    score_3_original
)

r_noun_p, p_noun_p = pearsonr(
    score_1_noun,
    score_3_noun
)

print(
    f"Original condition - "
    f"Pearson r: {r_orig_p}, P-Value: {p_orig_p}"
)

print(
    f"Noun condition - "
    f"Pearson r: {r_noun_p}, P-Value: {p_noun_p}"
)

