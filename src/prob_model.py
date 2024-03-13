import numpy as np

from sklearn.linear_model import LogisticRegressionCV

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import List

from conformal import compute_conformity_scores

def fit_model(
        features : np.ndarray,
        labels : np.ndarray, 
        config : dict,
        dataset_train : List = None,
        eval_dict : dict = None
):
    name = config.model.prob.name
    if name == "logistic":
        model = LogisticRegressionCV()
        model.fit(X=features, y=labels)
        return model
    elif name == "XGBoost":
        raise ValueError("not implemented yet")
    elif name == "torch":
        # no data splitting for now when constructing conformal loss
        model = LogisticRegression(features.shape[1])

        optimizer = optim.Adam(model.parameters(), lr=1)
        x = torch.tensor(features, requires_grad=True, dtype=torch.float32)

        for i in range(500):
            optimizer.zero_grad()
            probs = model.forward(x)

            loss, avg_train = get_conformal_loss(probs, labels, dataset_train, config.conformal.alpha)
            if i % 100 == 0:
                probs_valid = model.forward(torch.tensor(eval_dict['X_valid'], dtype=torch.float32)).detach().numpy()
                probs_split = np.array_split(probs_valid, eval_dict['splits_valid'])
                threshold = np.quantile(compute_conformity_scores(eval_dict['dataset_valid'], probs_split), 1 - config.conformal.alpha)
                probs_test = model.forward(torch.tensor(eval_dict['X_test'], dtype=torch.float32)).detach().numpy()
                probs_split = np.array_split(probs_test, eval_dict['splits_test'])
                avg = 0
                for prob in probs_split:
                    avg_retain = np.mean(prob > threshold.item())
                    avg += avg_retain
                print(f"Average % of train claims retained: {avg_train}")
                print(f"Average % of test claims retained: {avg / len(probs_split)}")
                print(f"Loss at iteration {i}: {loss.item()}")

            loss.backward()
            optimizer.step()
        return model

    else:
        return ValueError(f"{name} not available.")


def get_conformal_loss(probs, labels, dataset_train, alpha):
    claim_splits = torch.tensor(
            np.cumsum([len(dat['atomic_facts']) for dat in dataset_train])[:-1]
    )

    claim_probs = torch.tensor_split(probs, claim_splits)
    claim_labels = np.array_split(1 - labels, claim_splits.numpy())
    scores = []
    for c_prob, c_label in zip(claim_probs, claim_labels):
        scores.append(c_prob[c_label].max()) # could replace this with element-wise multiplication and make annotations softer?

    # use random set of scores to calibrate
    random_indices = np.random.permutation(len(scores))
    threshold_indices = random_indices[:25]
    loss_indices = random_indices[25:]

    threshold_scores = [scores[i] for i in range(len(scores)) if i in threshold_indices]
    
    threshold = torch.quantile(torch.stack(threshold_scores), 1 - alpha)
    loss = 0
    avg = 0
    for idx, c_prob in enumerate(claim_probs):
        if idx in loss_indices:
            loss += torch.sigmoid((threshold - c_prob)).mean()
            avg_retain = (c_prob > threshold).float().mean()
            avg += avg_retain
    if np.isnan(loss.item()):
        raise ValueError(claim_probs[0])
    return loss, avg / len(loss_indices)
    
class LogisticRegression(nn.Module):

    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return F.sigmoid(self.linear(x))
    
    
