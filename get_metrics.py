import pandas as pd
import json

print("\n--- Final Risk Scores ---")
df = pd.read_csv('risk_scores_v2.csv')
print(df['label_source'].value_counts())

print("\n--- EVT Thresholds ---")
with open('evt_thresholds_v2.json', 'r') as f:
    evt = json.load(f)
for k, v in evt.items():
    print(f"{k}: Threshold = {v['threshold']:.4f} (q={v['q']})")

print("\n--- Pseudo-Labels (Self-Training) ---")
with open('pseudo_labels_v2.json', 'r') as f:
    pseudo = json.load(f)
positives = len(pseudo['positive_set'])
negatives = len(pseudo['negative_set'])
rounds = pseudo['round']
print(f"Total Fraud Positives Promoted: {positives}")
print(f"Total Negatives (Remaining): {negatives}")
print(f"Self-Training Rounds Completed: {rounds}")

print("\n--- Sample Explanations (Top 3) ---")
with open('explanation_cards_v2.json', 'r') as f:
    cards = json.load(f)
for card in cards[:3]:
    print(f"\nApp ID: {card['application_id']}")
    print(f"Risk Score: {card['risk_score']:.4f} (from {card['label_source']})")
    print(f"Primary Channel: {card['anomaly_channel']}")
    print(f"Top Features: {', '.join(card['top_shap_features'][:3])}")
    print(f"Narrative: {card['narrative']}")
