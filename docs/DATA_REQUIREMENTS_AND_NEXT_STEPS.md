# Supervisor Requests: Pathway to a Production-Grade Evaluation

While the current 3-stage Machine Learning architecture is mathematically sound and successfully operational, it is currently operating under restricted dataset constraints. To transition this project from a strong "Proof of Concept" into a production-grade fraud detection system with a concrete evaluation, the following data requests should be addressed with the project supervisor.

They are ranked in order of critical importance.

---

### 1. Request a "Full State" or "Full Year" Data Dump
**The Problem:** Right now, the model was built and evaluated on a sample of 15,000 records. Scholarship fraud (especially identity duplication) is a relational crime. As seen in the initial evaluation, the 4 known fraud cases in our dataset failed to trigger duplicate identity rules because their "duplicate partners" were missing from our 15k sample. 
**The Ask:** Request a complete, uncut dataset for a specific state or specific academic year. The script requires the entire ecosystem of applications to accurately identify duplicate IPs, shared mobile networks, and duplicate identities across the whole network.

### 2. Request Historical, Human-Audited Ground Truth Labels
**The Problem:** The current model is evaluated against "Weak Labels" (applications that broke static rules). While our PR-AUC is high (0.76+), this evaluation metric primarily proves that the ML model is capable of learning the static rules.
**The Ask:** Request a dataset containing applications that were historically investigated and **confirmed** as fraud by human auditors at the NIC (or cases where scholarship funds had to be recovered). This provides a true "Ground Truth" target to evaluate the VAE's anomaly detection against real-world, verified fraud.

### 3. Request the Missing Financial Columns (Even if Hashed)
**The Problem:** Almost 30% of the NIC's static revalidation rules (Rules `K`, `H`, `V`, `W`, etc.) check for financial anomalies, such as multiple students routing funds to the same bank account or utilizing invalid IFSC codes. The current working CSV entirely omits the `bank_account_no`, `bank_name`, and `ifsc_code` columns.
**The Ask:** Request the inclusion of these financial columns. If applicant privacy is a concern, ask the database administrators to provide **cryptographically hashed** versions of the bank accounts (e.g., `SHA-256`). A hashed bank account still empowers our model to identify if 50 students are using the exact same account, without exposing the actual sensitive account numbers.

### 4. Request Access to the AISHE/DISE Master Database
**The Problem:** Several advanced revalidation rules (`UA` through `UI`) require cross-referencing the applicant's stated institution against the official Ministry of Education registries (AISHE for colleges, UDISE for schools). This is necessary to verify if the school legally exists, or if the number of scholarship applications has suspiciously exceeded the school's total physical enrollment capacity.
**The Ask:** Request a static data dump of the AISHE/DISE database. This external data can be integrated into `feature_selection.py` to automatically engineer features that flag "ghost institutions" or flagrant enrollment breaches.
