import pickle as pkl
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np


class MetricsDict(TypedDict):
    ant_location_metrics: dict[Any, list[tuple[float, float]]]
    returns: list[float]


class Metrics:
    def __init__(self):
        self.metrics: dict[str, MetricsDict] = defaultdict(MetricsDict)

    def save_metrics(self, file: str = "metrics.pkl"):
        with open(file, "wb") as f:
            pkl.dump(self.metrics, f)

    def load_metrics(self, file: str = "metrics.pkl"):
        with open(file, "rb") as f:
            self.metrics = pkl.load(f)

    def save_ant_location(self, algoname: str, skill: Any, x: float, y: float):
        if skill not in self.metrics[algoname]["ant_location_metrics"]:
            self.metrics[algoname]["ant_location_metrics"] = {}
            self.metrics[algoname]["ant_location_metrics"][skill] = []
        self.metrics[algoname]["ant_location_metrics"][skill].append((x, y))

    def plot_ant_explore_metrics(
        self, algoname: str, file: str = None, custom_title: str = None
    ):
        if "ant_location_metrics" not in self.metrics[algoname]:
            print(f"No ant location metrics found for {algoname}")
            return

        plt.figure(figsize=(8, 8))

        # n_skills = len(self.metrics[algoname]["ant_location_metrics"])

        for i, (skill, locations) in enumerate(
            self.metrics[algoname]["ant_location_metrics"].items()
        ):
            x, y = zip(*locations)
            plt.plot(x, y, label=f"skill {i}-th")

        plt.title(custom_title or f"{algoname} Ant Explore Metrics")
        # plt.xlabel('x')
        # plt.ylabel('y')
        plt.legend()
        if file:
            plt.savefig(file)
        plt.show()

    def save_return(self, algoname: str, ret: float):
        """
        Use this for downstream tasks, where we only care about the return of the final policy.
        Use at the end of episode, you should run this for a certain number of episodes -> this will save and later can print the mean and std of the return.
        """

        self.metrics[algoname]["returns"] = self.metrics[algoname].get(
            "returns", []
        ) + [ret]

    def print_return_metrics(self, algoname: str):
        returns = self.metrics[algoname].get("returns", [])
        if not returns:
            print(f"No return metrics found for {algoname}")
            return
        mean_return = sum(returns) / len(returns)
        std_return = (
            sum((r - mean_return) ** 2 for r in returns) / len(returns)
        ) ** 0.5
        print(
            f"{algoname} Return Metrics: Mean = {mean_return:.2f}, Std = {std_return:.2f}"
        )

    def plot_traj_confusion_matrix(
        self, encoder: Callable, *traj, file: str = None, custom_title: str = None
    ) -> float:
        """
        traj is list of observations
        encoder can turn each traj into an embedding space -> calc confusion matrix of the embeddings -> calculate the diff metric based on the confusion matrix
        """
        # support either a single iterable of observations or multiple obs args
        if len(traj) == 1:
            maybe_iter = traj[0]
            # if a single observation was passed (not iterable of obs), wrap it
            if not hasattr(maybe_iter, "__iter__") or isinstance(
                maybe_iter, (str, bytes)
            ):
                observations = [maybe_iter]
            else:
                observations = list(maybe_iter)
        else:
            observations = list(traj)

        embeddings = [encoder(obs) for obs in observations]
        if len(embeddings) == 0:
            print("No trajectory observations provided")
            return 0.0

        # convert embeddings to numpy array (n_samples, n_features)
        emb_arr = np.asarray(embeddings)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(-1, 1)

        # compute cosine similarity matrix without sklearn
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        normalized = emb_arr / norms
        conf_mat = normalized @ normalized.T

        # clip numerical noise to [-1,1]
        conf_mat = np.clip(conf_mat, -1.0, 1.0)

        plt.figure(figsize=(6, 6))
        im = plt.imshow(conf_mat, interpolation="nearest", cmap="viridis")
        plt.title(custom_title or "Trajectory Embedding Confusion Matrix")
        plt.xlabel("Time step")
        plt.ylabel("Time step")
        plt.colorbar(im)
        plt.tight_layout()
        if file:
            plt.savefig(file)
        plt.show()

        # return mean similarity as a simple metric
        return float(np.mean(conf_mat))
