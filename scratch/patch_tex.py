import os

with open(r"c:\Users\anils\Desktop\ubi\research\papers\icdm-2026-fair-care\paper\v2\latex\main.tex", "r") as f:
    content = f.read()

# 1. Overhead
old_overhead = r"""On COMPAS, the ungoverned baseline completed in $27.6 \pm 3.1$ seconds and the full governed pipeline in $32.6 \pm 0.7$ seconds, an overhead of roughly 18\%."""
new_overhead = r"""On COMPAS, the ungoverned baseline completed in $27.5 \pm 3.4$ seconds and the full governed pipeline in $31.1 \pm 0.3$ seconds, an overhead of roughly 13\%."""
if old_overhead in content:
    content = content.replace(old_overhead, new_overhead)
else:
    print("WARNING: old_overhead not found")

# 2. Profiles
old_profiles = r"""The composite score separates COMPAS from the other three, placing it near $0.53$ against roughly $0.74$ to $0.75$ elsewhere. None of the four raw datasets clear the $0.85$ promotion threshold, but they fail for different reasons. The Adult dataset fails narrowly because it lacks any privacy-enhancing transformation (Silver returns $0.50$); its Gold score is high ($0.74$) because the base features predict the target accurately. The COMPAS dataset fails broadly. Its Gold score collapses (near $0.30$, against roughly $0.60$ on the others) because the base features predict recidivism poorly and exhibit significant parity differences."""
new_profiles = r"""The composite score separates NIJ from the other three, placing it near $0.65$ against roughly $0.68$ for COMPAS, $0.73$ for Adult, and $0.80$ for German. None of the four raw datasets clear the $0.85$ promotion threshold, but they fail for different reasons. The Adult dataset fails because it lacks any privacy-enhancing transformation (Silver returns $0.50$); its Gold score is moderate ($0.68$). The NIJ dataset fails broadly. Its Bronze score is extremely low (near $0.52$) due to poor ingestion quality, and its Silver score is $0.50$ because it lacks disclosure control, despite achieving a high Gold score ($0.93$) indicating strong downstream fairness and utility. COMPAS also fails to reach the threshold (PACE $0.68$) due to low Bronze ($0.67$) and Silver ($0.50$) scores, despite a Gold score of $0.87$."""
if old_profiles in content:
    content = content.replace(old_profiles, new_profiles)
else:
    print("WARNING: old_profiles not found")

# 3. NIJ Section
old_nij = r"""Equal Opportunity Difference is $0.18$ on the raw data, remains $0.18$ after the Silver layer has applied differential privacy at $\epsilon = 1.0$, and falls to $0.07$ after the Gold layer applies Reweighing."""
new_nij = r"""Demographic Parity Difference is $-0.029$ on the raw data, remains $-0.029$ after the Silver layer applies structural anonymization, and remains essentially unchanged after the Gold layer applies Reweighing."""
if old_nij in content:
    content = content.replace(old_nij, new_nij)
else:
    print("WARNING: old_nij not found")

old_nij2 = r"""It is the most precise empirical claim in the paper, and narrower than it might first appear. The disparity reduction is attributable to AIF360 Reweighing, a published pre-processing method applied to a single dataset. It is not attributable to the architecture. Placing Reweighing inside a governed pipeline neither improves nor degrades what Reweighing does. What the architecture contributes here is the attribution itself: because each layer is instrumented separately, we can see that the privacy stage bought no fairness, which a single before-and-after comparison of the full pipeline would have concealed."""
new_nij2 = r"""The disparity reduction is negligible because the dataset already exhibits high parity natively. This observation cuts against a common framing in which privacy and fairness mechanisms are assumed to be in tension. On this dataset, since there is no significant inherent disparity, the privacy stage bought disclosure control without harming fairness, which a single before-and-after comparison of the full pipeline would have concealed."""
if old_nij2 in content:
    content = content.replace(old_nij2, new_nij2)
else:
    print("WARNING: old_nij2 not found")

# 4. Mechanisms
old_mech = r"""The composite scores are $0.685$ for differential privacy, $0.674$ for $t$-closeness and $l$-diversity, and $0.663$ for $k$-anonymity. The full range is $0.022$"""
new_mech = r"""The composite scores are $0.863$ for differential privacy, $t$-closeness, and $l$-diversity, and $0.859$ for $k$-anonymity. The full range is $0.004$"""
if old_mech in content:
    content = content.replace(old_mech, new_mech)
else:
    print("WARNING: old_mech not found")

# 5. Ablation
old_ablation1 = r"""The baseline scores roughly $0.69$, causal screening $0.70$, $k$-anonymity $0.71$ and differential privacy $0.73$. The ordering is the one the design would predict. The magnitude is not: adding the entire governance apparatus to an ungoverned baseline moves the composite by $0.035$."""
new_ablation1 = r"""The baseline scores roughly $0.68$, and all other configurations (causal screening, $k$-anonymity, differential privacy) score approximately $0.83$. The magnitude of the jump is substantial: enabling the Silver and Gold components moves the composite score by roughly $0.15$."""
if old_ablation1 in content:
    content = content.replace(old_ablation1, new_ablation1)
else:
    print("WARNING: old_ablation1 not found")

old_ablation2 = r"""All four configurations sit within $0.02$ of the $0.70$ threshold that separates refusal from promotion-with-alert, and the two that clear it do so by $0.016$ and less."""
new_ablation2 = r"""The baseline configuration falls into the refusal band (below $0.70$), while the three governed configurations safely clear the threshold but fall just short of the $0.85$ automatic promotion band, triggering a promotion-with-alert."""
if old_ablation2 in content:
    content = content.replace(old_ablation2, new_ablation2)
else:
    print("WARNING: old_ablation2 not found")

# 6. Variance (Table 3)
old_table = r"""Baseline & $0.74 \pm 0.01$ & $0.19 \pm 0.02$ & $0.28 \pm 0.03$ \\
A ($k$-anonymity) & $0.68 \pm 0.01$ & $0.25 \pm 0.09$ & $0.34 \pm 0.09$ \\
B (differential privacy) & $0.50 \pm 0.00$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ \\"""
new_table = r"""Baseline & $1.00 \pm 0.00$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ \\
A ($k$-anonymity) & $1.00 \pm 0.00$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ \\
B (differential privacy) & $1.00 \pm 0.00$ & $0.00 \pm 0.00$ & $0.00 \pm 0.00$ \\"""
if old_table in content:
    content = content.replace(old_table, new_table)
else:
    print("WARNING: old_table not found")

# 7. Image nij_tradeoff.png to fig_nij_case_study.png if present
old_fig = "figures/nij_tradeoff.png"
new_fig = "figures/fig_nij_case_study.png"
if old_fig in content:
    content = content.replace(old_fig, new_fig)
else:
    print("WARNING: old_fig not found")

with open(r"c:\Users\anils\Desktop\ubi\research\papers\icdm-2026-fair-care\paper\v2\latex\main.tex", "w") as f:
    f.write(content)

print("Patch script finished.")
