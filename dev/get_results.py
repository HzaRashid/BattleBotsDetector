#!/usr/bin/env python3
import json
import os
import argparse



def evaluate_detector(results_path, detector_name="hamzadetector1"):
    # Load the JSON file
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter only those entries that have both ground truth and detectors
    entries = data["users"]
    true_count = 0
    false_count = 0

    tp = fp = tn = fn = 0
    for rec in entries:
        true_label = bool(rec["is_bot"])
        true_count += rec["is_bot"]
        false_count += not rec["is_bot"]
        # find this detector's prediction
        det = next((d for d in rec["detectors"] 
                    if d.get("teamName") == detector_name), None)
        if det is None:
            # detector missing for this record; skip
            continue
        pred_label = bool(det.get("isBot", False))

        if pred_label and true_label:
            tp += 1
        elif pred_label and not true_label:
            fp += 1
        elif not pred_label and not true_label:
            tn += 1
        elif not pred_label and true_label:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


    print(f"Results for {os.path.basename(results_path)}:")
    # print(f"Detector: {detector_name}")
    # print(f"Total evaluated: {total}")
    # print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Precision (bot): {precision:.4f}")
    # print(f"Recall    (bot): {recall:.4f}")
    # print(f"F1-score  (bot): {f1:.4f}")

    print('human to bot ratio:', false_count/true_count)
    return false_count/true_count

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Evaluate overall accuracy and bot‐class precision/recall/F1 for a given detector"
    # )
    # parser.add_argument("results_file", help="Path to JSON results file")
    # parser.add_argument(
    #     "--detector", "-d", default="hamzadetector1",
    #     help="Name of detector in the 'teamName' field (default: hamzadetector1)"
    # )
    # args = parser.parse_args()

    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, './data')
    # session_num = 16
    # session_path = os.path.join(data_dir, f'session_{session_num}_results.json')
    # evaluate_detector(session_path, 'hamzadetector1')

    sessions = [
        # 11, 
                12, 13, 14, # random forest
                16, 17, 18, 19, 20, 21 # multimodal
                ]
    ratios = 0
    for sess in sessions:
        session_path = os.path.join(data_dir, f'session_{sess}_results.json')
        ratios += evaluate_detector(session_path, 'hamzadetector1')
        print()

    print(ratios/len(sessions))