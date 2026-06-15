# Data Analysis and Findings Report: Feature Profile and Static Rules Mapping

This document provides a comprehensive report of the findings from the data analysis performed on the NIC Fraud Detection project datasets. The goal of this analysis is to profile the features in the primary applicant dataset, evaluate their relationship with the existing static revalidation rules, identify structural data issues, and outline recommendations for a future Machine Learning model.

---

## 1. Executive Summary

1. **Clean Baseline Data**: The primary dataset `data_for_ml_model.csv` (15,000 records) contains almost exclusively valid cases. Only **4 records (0.027%)** are flagged as invalid (violating revalidation rules) in the `sanity` column. 
2. **Missing Key Indicators**: Several vital fraud-related fields referenced in the `Revalidation.xlsx` rules—specifically bank account numbers, bank names, and IFSC codes—are **entirely missing** from the CSV. This presents a critical gap for reproducing or modeling rules related to bank account sharing and IFSC consistency.
3. **High Spatial and Attribute Redundancy**: The dataset contains multiple duplicate location columns (such as redundant state and district IDs/names) and **15 columns that are 100% null**. Cleaning these features will be essential before modeling.
4. **Calculated Temporal Metrics**: The applicant's age at the time of registration is a high-utility engineered feature that is strongly correlated with the age limit boundaries defined in the static rules.

---

## 2. Primary Dataset Overview (`data_for_ml_model.csv`)

- **Shape**: 15,000 rows × 136 columns.
- **Data Completeness**:
  - **15 Columns are 100% null**: `updated_by`, `delete_record`, `deleted_by`, `delete_on`, `delete_ip_address`, `deleted_by_level`, `c_university_id`, `p_institution_id`, `x_institution_id`, `xii_institution_id`, `competitive_exam_score`, `xii_course_id`, `new_entitled_fee_amount_centre_share`, `sub_category_id`, `updated_by-2`, and `updated_on-2`.
  - **High Nullity Fields (>95% null)**: 
    - `sanity` (99.97% null - indicating valid records)
    - `disability_percentage` & `disablity_type` (99.49% null - expected as disability is rare)
    - `orphan_flag` (99.75% null)
    - `gaurdian_name` (99.77% null)
    - `enroll_udid_no` (99.49% null)
    - `ration_card_no` & `ration_card_member_no` (96.49% null)
    - `district_short_name` (99.97% null)

---

## 3. Revalidation Rules Analysis (`Revalidation.xlsx`)

The static rules sheet (`Sheet1`) contains **99 unique rules** currently used by the evaluation system. We categorized these rules into six key domains:

| Category | Description | Examples of Rules |
| :--- | :--- | :--- |
| **1. Duplicate Checks** | Flags duplicate identities by matching name, DOB, mobile, Aadhaar, Class 10/12 roll numbers, or bank accounts. | `A`, `B`, `F`, `G`, `M`, `H`, `R1-R6`, `X9`, `X10`, `Z`, `V`, `W`, `A1`, `R8`, `A2`, `R9`, `YK`, `YL`, `YO`, `YP` |
| **2. Location Consistency** | Checks for geographical mismatches between applicant's domicile, institute location, and bank branch location. | `X4`, `X5`, `UU`, `YC`, `YD`, `YF`, `SD` |
| **3. Age Boundaries** | Standard age limits matching the Pre-Matric (age ≤ 20) and Post-Matric (13 ≤ age ≤ 35) eligibility criteria. | `X1`, `X7`, `X8`, `X16` |
| **4. Financial Thresholds** | Flags applications with family income under standard thresholds (e.g., < 20,000 or ≤ 10,000). | `UW`, `X13`, `X21` |
| **5. Institute Concentration** | Monitors aggregate volume patterns per institute (e.g., enrollment exceeded, gender skewness, excessive hostellers). | `UA-UG`, `UI`, `UH`, `UB`, `UC`, `UV`, `UX`, `UY`, `UT` |
| **6. Integration Verification** | Flags applications that fail external API validations (e.g., UDID validation, AICTE master list mismatches). | `VA`, `C`, `M1`, `M9`, `M10`, `A3` |

---

## 4. Feature Group Findings

### 4.1 Demographic & Age Analysis
- **Fresh vs. Renewal**: All 15,000 applicants are labeled as **Fresh (`F`)**.
- **Age Calculations**: We computed the applicant's age at the time of registration (`registered_date` - `date_of_birth`):
  - **Pre-Matric Scheme (pre_post_matric = 1)**: 5,073 applicants. Age ranges from **5.56 to 19.11 years** (Mean: 13.84). No pre-matric applicants exceed the 20-year limit (Rule `X1`).
  - **Post-Matric Scheme (pre_post_matric = 2)**: 9,908 applicants. Age ranges from **13.14 to 40.21 years** (Mean: 18.66). 
  - **Rule X7 Compliance**: **3 post-matric applicants exceed the 35-year limit** (Ages: 35.8, 38.3, and 40.2), yet they were not flagged under `sanity`. This represents a potential gap in current rule enforcement.

### 4.2 Contact & Address Details
- **Mobile Number Sharing**:
  - 14,867 mobile numbers appear exactly once.
  - 59 mobile numbers are shared by exactly 2 applicants.
  - 3 mobile numbers are shared by exactly 3 applicants.
  - 1 mobile number is shared by 6 applicants.
  - **No extreme sharing** (e.g., >10 or >20 applications under rules `YK` and `YL`) was observed.
- **IP Address Sharing**: There is a high concentration of registrations from specific IP addresses. The top IP address submitted **31 applications**, and several others submitted 10-20 applications. This indicates registrations done at common locations (e.g., cyber cafes or schools).

### 4.3 Location Fields and Spatial Redundancies
The dataset contains high spatial redundancy. The following fields are 100% duplicate:
- `domicile_state_id` == `state_id` == `state_id-2` == `pfms_state_code`
- `state_name` == `state_name-2`
- `permanent_district_id` == `district_id`
- `district_name` == `district_name-2`

### 4.4 Academic & Course Information
- **Mode of Study**: 14,992 applicants (99.95%) are in regular mode (`modeofstudy` = 1). 6 are in distance mode, and 2 are in part-time mode.
- **Institute Concentrations**:
  - The top institute (`c_institution_id` = 13862) has **161 applications**.
  - No institute exceeds 500 applications (Rule `X3` limit is 500).

### 4.5 Financial Profiles
- **Tuition Fee**: Mean is 15,223 INR; max is 100,000 INR.
- **Family Income**: 
  - Mean family income is 110,657 INR (Max: 4,000,000 INR).
  - **137 applicants** report family income < 20,000 (Rule `UW`).
  - **44 applicants** report family income ≤ 10,000 (Rules `X13` / `X21`).
  - *Anomaly*: Some records show a family income as low as 5 INR. This likely represents input placeholders or incorrect data entry.

---

## 5. Coverage Gap Analysis (Rules vs. Features)

### 5.1 Evaluable Rules (Using CSV columns)
The following rules can be evaluated directly using the columns in the CSV:
- **Demographic Dupes** (`A1`, `A2`, `A`, `B`, `F`, `G`, `M`, `R1-R6`): Uses names, DOB, and mobile numbers.
- **Age Bounds** (`X1`, `X7`, `X8`): Uses DOB and scheme type.
- **Income Limits** (`UW`, `X13`, `X21`): Uses `annual_family_income`.
- **Identity Matches** (`YF`): Compares `applicant_name` with `father_name` and `mother_name`.

### 5.2 Unevaluable Rules (Missing Data)
These rules cannot be evaluated because the necessary data is missing from the CSV:
- **Bank Account Rules** (`K`, `H`, `X2`, `X4`, `X5`, `V`, `W`): The dataset does not contain bank name, account number, or IFSC.
- **External Database Matches** (`UA-UI`): Requires external institute capacity strengths (AISHE/DISE enrollment counts) mapped to `c_institution_id`.

---

## 6. Recommendations for Machine Learning

### 6.1 Modeling Strategy
Because the dataset is **99.97% valid cases**, training a standard supervised binary classifier (e.g. Random Forest, SVM) directly will fail due to extreme class imbalance. We recommend two distinct modeling paths:

1. **Path A: Unsupervised Anomaly Detection (Recommended)**
   - **How it works**: Use algorithms like **Isolation Forest**, **One-Class SVM**, or **Autoencoders** trained *only* on the valid records.
   - **Benefit**: These models learn the statistical structure of "normal" applications. When an application deviates significantly (e.g., anomalous age-income ratio or suspicious concentrations), it is flagged. This enables the model to detect **new, unseen fraud patterns** that static rules might miss.
2. **Path B: Rule-Synthesized Supervised Learning**
   - **How it works**: Programmatically run the static rules on the 15,000 records to identify any hidden violations, or synthetically inject synthetic anomalies matching the rules. Use the resulting labeled dataset to train a tree-based supervised classifier (e.g., **XGBoost** or **LightGBM**).
   - **Benefit**: A tree-based supervised model is highly efficient at learning and mimicking complex multi-dimensional boundary conditions (like the combinations of age, income, and schemes).

### 6.2 Suggested Engineered Features
To improve the ML model's ability to detect anomalous applications, the following engineered features should be constructed:
- **`age_at_registration`**: Continuous variable derived from DOB and registration date.
- **`state_match_flag`**: Binary indicator showing whether the applicant's domicile state matches the institution's state.
- **`mobile_occurrence_count`**: Frequency count of how many times the applicant's mobile number appears in the dataset.
- **`ip_occurrence_count`**: Frequency count of registrations from the same IP address.
- **`fee_income_ratio`**: Ratio of total fees (tuition + admission + misc) to the annual family income, which can flag cases where tuition fees are disproportionately high relative to reported income.
