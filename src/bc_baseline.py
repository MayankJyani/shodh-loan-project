import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_npz(path):
    d = np.load(path)
    return d['observations'], d['actions'].squeeze(), d['rewards']

def evaluate_policy(actions, rewards):
    # expected total reward and avg reward per step
    total = (actions * rewards).sum() + ((1 - actions) * 0).sum()
    avg = total / len(rewards)
    return total, avg

if __name__ == "__main__":
    out = Path("../data/rl")
    train_obs, train_act, train_rew = load_npz(out / "train_rl.npz")
    test_obs, test_act, test_rew = load_npz(out / "test_rl.npz")

    # standardize features
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_obs)
    test_X = scaler.transform(test_obs)

    # train BC (predict action from state)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(train_X, train_act)

    pred_test = clf.predict(test_X)
    print("Test classification metrics (BC vs historical actions):")
    print("  Acc:", accuracy_score(test_act, pred_test))
    print("  Prec:", precision_score(test_act, pred_test))
    print("  Rec:", recall_score(test_act, pred_test))
    print("  F1:", f1_score(test_act, pred_test))

    # Evaluate policy by computing rewards when policy selects action=1 (approve)
    total_reward, avg_reward = evaluate_policy(pred_test, test_rew)
    print(f"\nPolicy reward on test set: total={total_reward:.2f}, avg_per_loan={avg_reward:.6f}")
    # also show historical policy reward (all approved = actions in file)
    hist_total, hist_avg = evaluate_policy(test_act, test_rew)
    print(f"Historical policy (from data) reward: total={hist_total:.2f}, avg_per_loan={hist_avg:.6f}")