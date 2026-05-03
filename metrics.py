import pickle as pkl
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde



class MetricsDict(TypedDict):
    x_y_location_metrics: dict[str, dict[Any, list[tuple[float, float]]]]    # key = env_name, value = dict of skill -> list of (x,y) locations
    x_location_metrics: dict[str, dict[Any, list[float]]]                     # key = env_name, value = dict of skill -> list of x locations
    returns: dict[str, list[float]]                         # key = env_name, value = list of returns


class Metrics:
    def __init__(self, load: bool = False, file: str = "metrics.pkl"):
        self.metrics: dict[str, MetricsDict] = self.__init_metrics()
        if load:
            self.load_metrics(file)

    def list_algorithms(self) -> list[str]:
        return list(self.metrics.keys())

    def remove_algorithm(self, algoname: str):
        if algoname in self.metrics:
            del self.metrics[algoname]

    def remove_env_from_algorithm(self, algoname: str, env_name: str):
        if algoname in self.metrics:
            if env_name in self.metrics[algoname]["x_y_location_metrics"]:
                del self.metrics[algoname]["x_y_location_metrics"][env_name]
            if env_name in self.metrics[algoname]["x_location_metrics"]:
                del self.metrics[algoname]["x_location_metrics"][env_name]
            if env_name in self.metrics[algoname]["returns"]:
                del self.metrics[algoname]["returns"][env_name]

    def list_metrics(self):
        ret = {}
        for algoname, metrics_dict in self.metrics.items():
            ret[algoname] = {}

            envs = list(metrics_dict["x_y_location_metrics"].keys())
            if envs:
                ret[algoname]["x_y_location_metrics"] = envs
            envs = list(metrics_dict["x_location_metrics"].keys())
            if envs:
                ret[algoname]["x_location_metrics"] = envs
            envs = list(metrics_dict["returns"].keys())
            if envs:
                ret[algoname]["returns"] = envs


        return ret

    @staticmethod
    def __init_metrics(data: dict[str, MetricsDict] | None = None) -> dict[str, MetricsDict]:
        metrics = defaultdict(lambda: {"x_y_location_metrics": {}, "x_location_metrics": {}, "returns": {}})

        if data is not None:
            for algoname, metrics_dict in data.items():
                metrics[algoname] = metrics_dict

        return metrics

    def save_metrics(self, file: str = "metrics.pkl"):
        with open(file, "wb") as f:
            pkl.dump(dict(self.metrics), f)

    def load_metrics(self, file: str = "metrics.pkl"):
        with open(file, "rb") as f:
            data = pkl.load(f)
            self.metrics = self.__init_metrics(data)

    def save_x_y_location(self, algoname: str, env_name: str, skill: Any, x: float, y: float):
        if env_name not in self.metrics[algoname]["x_y_location_metrics"]:
            self.metrics[algoname]["x_y_location_metrics"][env_name] = {}

        if skill not in self.metrics[algoname]["x_y_location_metrics"][env_name]:
            self.metrics[algoname]["x_y_location_metrics"][env_name][skill] = []

        self.metrics[algoname]["x_y_location_metrics"][env_name][skill].append((x, y))

    def plot_x_y_explore_metrics(
        self, algoname: str, env_name: str, cmap_name = 'hsv', file: str = None, custom_title: str = None, legend: bool = False, x_lim = (-10, 10), y_lim = (-10, 10)
    ):
        if "x_y_location_metrics" not in self.metrics[algoname]:
            print(f"No x_y location metrics found for {algoname}")
            return

        if env_name not in self.metrics[algoname]["x_y_location_metrics"]:
            print(f"No x_y location metrics found for {algoname} in environment {env_name}")
            return

        plt.figure(figsize=(8, 8))

        n_skills = len(self.metrics[algoname]["x_y_location_metrics"][env_name])  # n_colors = n_skills
        # colors = plt.cm.get_cmap("tab20", max(n_skills, 1))
        cmap = plt.cm.get_cmap(cmap_name, n_skills)  # discrete version

        colors = cmap(np.arange(n_skills))

        for i, (skill, locations) in enumerate(
            self.metrics[algoname]["x_y_location_metrics"][env_name].items()
        ):
            x, y = zip(*locations)
            plt.plot(x, y, label=f"skill {i}-th", color=colors[i])

        plt.title(custom_title or f"{algoname} {env_name} Explore Metrics")
        # plt.xlabel('x')
        # plt.ylabel('y')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        if legend:
            plt.legend()
        if file:
            plt.savefig(file)
        plt.show()

    def save_x_location(self, algoname: str, env_name: str, skill: Any, x: float):
        if env_name not in self.metrics[algoname]["x_location_metrics"]:
            self.metrics[algoname]["x_location_metrics"][env_name] = {}

        if skill not in self.metrics[algoname]["x_location_metrics"][env_name]:
            self.metrics[algoname]["x_location_metrics"][env_name][skill] = []

        self.metrics[algoname]["x_location_metrics"][env_name][skill].append(x)

    def plot_x_location_metrics(
        self, algoname: str, env_name: str, cmap_name = 'hsv', file: str = None, custom_title: str = None, legend: bool = False, x_lim = (-10, 10)
    ):
        if "x_location_metrics" not in self.metrics[algoname]:
            print(f"No x location metrics found for {algoname}")
            return

        if env_name not in self.metrics[algoname]["x_location_metrics"]:
            print(f"No x location metrics found for {algoname} in environment {env_name}")
            return

        plt.figure(figsize=(8, 8))

        n_skills = len(self.metrics[algoname]["x_location_metrics"][env_name])  # n_colors = n_skills
        # colors = plt.cm.get_cmap("tab20", max(n_skills, 1))
        cmap = plt.cm.get_cmap(cmap_name, n_skills)  # discrete version

        colors = cmap(np.arange(n_skills))

        for i, (skill, locations) in enumerate(
            self.metrics[algoname]["x_location_metrics"][env_name].items()
        ):
            x = np.asarray(locations)
            y = i
            # plt.plot(x, y, label=f"skill {i}-th", color=colors[i])
            kde = gaussian_kde(x)

            xs = np.linspace(x.min(), x.max(), 500)

            density = kde(xs)

            density = (density - density.min()) / density.max()

            plt.fill_between(xs, y*2 + density, y*2 - density, color=colors[i], alpha=0.25)

        plt.title(custom_title or f"{algoname} {env_name} X Location Metrics")
        # plt.xlabel('Time step')
        # plt.ylabel('X position')
        plt.yticks([])
        plt.xlim(x_lim)
        if legend:
            plt.legend()
        if file:
            plt.savefig(file)
        plt.show()

    def save_return(self, algoname: str, env_name: str, ret: float):
        """
        Use this for downstream tasks, where we only care about the return of the final policy.
        Use at the end of episode, you should run this for a certain number of episodes -> this will save and later can print the mean and std of the return.
        """

        self.metrics[algoname]["returns"][env_name] = self.metrics[algoname].get(
            "returns", {}
        ).get(env_name, []) + [ret]

    def print_return_metrics(self, algoname: str, env_name: str):
        returns = self.metrics[algoname].get("returns", {}).get(env_name, [])
        if not returns:
            print(f"No return metrics found for {algoname} in environment {env_name}")
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
