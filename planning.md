# Planning Specification: TakeMeter (r/AmItheAsshole)

## 1. Community
The chosen community is **r/AmItheAsshole**, a massive online forum where users describe personal conflicts, and commenters vote on who is at fault. This community is a perfect fit for a text classification task because the discourse is highly structured, text-heavy, and relies on distinct, community-defined acronym signals to pass judgment.

## 2. Labels
* **NTA (Not the Asshole):** The commenter judges that the original poster (OP) was in the right and the other party involved in the conflict was entirely at fault.
  * *Example 1:* "NTA. You set clear boundaries months ago, and they completely ignored them. You have nothing to feel guilty about."
  * *Example 2:* "Definitively NTA. Your sister is acting incredibly entitled here; you are under no obligation to finance her vacation."
* **YTA (You're the Asshole):** The commenter judges that the original poster was wrong, selfish, or acting unreasonably in the situation described.
  * *Example 1:* "YTA. You went through his private journal without permission. That is a massive breach of trust, no matter how curious you were."
  * *Example 2:* "YTA. You sound incredibly defensive. You ruined her big moment just because you couldn't stand not being the center of attention."
* **ESH (Everyone Sucks Here):** The commenter judges that both the original poster and the other parties involved acted poorly, and no one is blameless.
  * *Example 1:* "ESH. Your brother shouldn't have borrowed your car without asking, but smashing his Xbox in revenge was an incredibly immature overreaction."
  * *Example 2:* "ESH. All of you need to learn how to communicate like adults instead of playing these passive-aggressive mind games."

## 3. Hard Edge Cases
* **The Ambiguous Boundary:** Comments where a user spends 90% of the text validating the poster's feelings ("I totally get why you were mad, they treated you terribly...") but concludes with a soft reprimand ("...but yeah, you shouldn't have screamed at them, so ESH").
* **Decision Rule:** The final legalistic acronym verdict dictates the label. If the text body contains conflicting sentiments but explicitly opens or closes with a definitive tag (e.g., "ESH"), it is assigned that tag. If no acronym is present, the label is determined by whether the text assigns moral fault exclusively to one party (NTA/YTA) or explicitly distributes blame to both sides (ESH).

## 4. Data Collection Plan & Final Distribution
Data was organically sourced from public comment sections across hot and trending conflict discussions within r/AmItheAsshole. While the original target was a strictly balanced class distribution (~70 comments per class), real-world community behavior yielded an organic, imbalanced social distribution across 245 total annotated examples:

* **NTA:** 132 comments (53.8%)
* **YTA:** 96 comments (39.2%)
* **ESH:** 17 comments (7.0%)

This real-world distribution was intentionally preserved in a 70/15/15 stratified split to test the limits of transformer-based transfer learning when confronting sparse minority class edge cases (such as mutual blame dynamics). No single label accounts for more than 70% of the dataset, satisfying the imbalance safeguard criteria.
## 5. Evaluation Metrics
* **Overall Accuracy:** To capture baseline system performance across the entire test set.
* **Per-Class F1-Score:** Crucial for this task to ensure the model isn't completely failing on the scarcer `ESH` class while hiding behind high accuracy on `NTA` or `YTA`. F1 balances precision and recall seamlessly.
* **Confusion Matrix:** To identify whether the model struggles with the directional boundary between partial blame (`ESH`) and total blame (`YTA`).

## 6. Definition of Success
The classifier will be deemed "good enough" for deployment as an automated community moderation or analytics tool if it achieves an **Overall Accuracy of >= 75%** and a **Per-Class F1-Score of >= 0.70** across all three categories on the test set.

## 7. AI Tool Plan
* **Label Stress-Testing:** Completed. Verified that explicit acronym anchors serve as effective ground truth resolution rules for mixed-sentiment text.
* **Annotation Assistance:** An LLM will be used to generate realistic synthetic training rows to vastly accelerate dataset bootstrapping, followed by human review to verify structural consistency.
* **Failure Analysis:** Misclassified rows will be piped to an AI evaluator to diagnose syntactic patterns (such as length or sentence structure) causing decision boundary errors.
