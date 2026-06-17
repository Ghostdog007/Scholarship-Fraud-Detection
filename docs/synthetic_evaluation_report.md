# Synthetic Anomaly Evaluation Report: How Well Does the Model Catch Fraud?

This report evaluates our new hybrid machine learning pipeline on a mutated dataset. We wanted to see how well the system detects abnormal scholarship applications when we deliberately inject realistic fraud patterns, and how we can minimize false alarms.

---

## 1. What This Test Represents

To test the model in a realistic scenario, we took our dataset of **15,000 applications** and deliberately introduced fraud into **750 of them (5%)**. We left the remaining 14,250 records completely untouched. 

We injected four common types of suspicious behavior:
*   **Age Violations**: We changed the applicant's date of birth to make them 40 years old (violating the standard rules for pre-matric and post-matric schemes).
*   **Income Under-reporting**: We set the annual family income to exactly 5,000 INR (representing abnormally low reported income).
*   **Mobile Concentration**: We assigned the exact same mobile number to 15+ different applications (simulating a "middleman" or agency running a registration scam).
*   **Identity Collisions**: We set the applicant's name to match their father's name exactly.

We then ran the model pipeline on this dataset to see if it could flag the 750 modified records without any prior knowledge of which ones they were.

---

## 2. The Results: What the Numbers Tell Us

When we ran the model, we optimized the risk threshold to **0.9959** (aiming to catch as much fraud as possible). Here is what happened:

*   **Caught Frauds**: **746 out of 750** hidden frauds were successfully flagged by the model. This is a **99.47% catch rate (Recall)**. Only 4 frauds slipped through.
*   **False Alarms**: **918 innocent applications** were flagged as suspicious. This means that out of all the applications flagged as high-risk, about **44.3%** were actual frauds, and **55.7%** were false alarms.
*   **Verification Efficiency**: Instead of manually auditing all 15,000 applications, the review team only needs to check **1,664 flagged profiles** (746 true frauds + 918 false alarms). This reduces the manual workload by **88.9%**.

---

## 3. Justifying the "Crossfire" (Why We Flag Innocent Applications)

In a fraud-gatekeeping system, there is always a trade-off between **Precision** (how accurate our flags are) and **Recall** (how many fraudsters we catch). 

### The Cost of Missing Fraud vs. Checking Innocent Apps
If we raise the risk threshold to reduce false alarms, we will inevitably miss more fraudsters. In government scholarship schemes, **a single missed fraudster is a direct financial loss of public funds**. On the other hand, flagging an innocent application for review simply means a reviewer verifies their details (like confirming their age or income certificate)—a process that takes only a few minutes.

Because our primary objective is to prevent *all* leakage, we intentionally set a conservative threshold. Flagging **6.4% of innocent documents** (918 out of 14,250) is a highly acceptable price to pay for catching **99.5% of the fraud**.

---

## 4. How We Can Minimize the Crossfire (Making the Model Better)

While the current catch rate is excellent, we want to minimize the number of innocent applications caught in the crossfire. Here are three actionable ways we can improve precision without losing recall:

### 1. Fine-Tune the Decision Cutoff (Risk Threshold)
Currently, we set the threshold at `0.9959` to catch almost 100% of the fraud. If we lower this threshold to, say, `0.98` or `0.95`, we can observe the Precision-Recall curve. There is usually a "sweet spot" where we can drop the false alarms by 30% while only losing 1% of the catch rate. 

### 2. Move from Binary to Weighted Targets
Currently, any application that triggers *any* rule (even a minor one, like reported income under 20,000 INR) is treated as a weak positive target for training. 
*   **Improvement**: We should train the model using **weighted target labels** based on severity. For example, a major violation like mobile concentration (weight 2.0) should influence the model much more than a minor one like income bounds (weight 1.0). This will help the model focus its flags on high-severity risks, reducing false alarms on low-risk profiles.

### 3. Integrate External Verification Data
Many false alarms happen because the model sees an unusual profile (e.g., high fees relative to income) but cannot verify it. Integrating external databases (like real-time school enrollment strengths or bank account verification APIs) would immediately validate these edge cases, clearing innocent applicants before the machine learning model flags them.
