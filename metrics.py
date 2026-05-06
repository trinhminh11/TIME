import pickle as pkl
from collections import defaultdict
from typing import Any, TypedDict

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_distances
from functools import lru_cache


class AliasDict(defaultdict):
    aliases = {"TIME": "TIME_64"}

    def __getitem__(self, key):
        if key in self.aliases:
            key = self.aliases[key]
        return super().__getitem__(key)


class MetricsDict(TypedDict):
    x_y_location_metrics: dict[
        str, dict[Any, list[tuple[float, float]]]
    ]  # key = env_name, value = dict of skill -> list of (x,y) locations
    x_location_metrics: dict[
        str, dict[Any, list[float]]
    ]  # key = env_name, value = dict of skill -> list of x locations
    trajectory_embeddings: dict[
        str, dict[Any, np.ndarray]
    ]  # key = env_name, value = dict of skill -> trajectory embedding values
    returns: dict[str, list[float]]  # key = env_name, value = list of returns


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
            if env_name in self.metrics[algoname]["trajectory_embeddings"]:
                del self.metrics[algoname]["trajectory_embeddings"][env_name]

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
            envs = list(metrics_dict["trajectory_embeddings"].keys())
            if envs:
                ret[algoname]["trajectory_embeddings"] = envs

        return ret

    @staticmethod
    def __init_metrics(
        data: dict[str, MetricsDict] | None = None,
    ) -> dict[str, MetricsDict]:
        metrics = AliasDict(
            lambda: {
                "x_y_location_metrics": {},
                "x_location_metrics": {},
                "returns": {},
                "trajectory_embeddings": {},
            }
        )

        if data is not None:
            for algoname, metrics_dict in data.items():
                if "trajectory_embeddings" not in metrics_dict:
                    metrics_dict["trajectory_embeddings"] = {}
                metrics[algoname] = metrics_dict

        return metrics

    def save_metrics(self, file: str = "metrics.pkl"):
        with open(file, "wb") as f:
            pkl.dump(dict(self.metrics), f)

    def load_metrics(self, file: str = "metrics.pkl"):
        with open(file, "rb") as f:
            data = pkl.load(f)
            self.metrics = self.__init_metrics(data)

    def save_x_y_location(
        self, algoname: str, env_name: str, skill: Any, x: float, y: float
    ):
        if env_name not in self.metrics[algoname]["x_y_location_metrics"]:
            self.metrics[algoname]["x_y_location_metrics"][env_name] = {}

        if skill not in self.metrics[algoname]["x_y_location_metrics"][env_name]:
            self.metrics[algoname]["x_y_location_metrics"][env_name][skill] = []

        self.metrics[algoname]["x_y_location_metrics"][env_name][skill].append((x, y))

    def plot_x_y_explore_metrics(
        self,
        algoname: str,
        env_name: str,
        cmap_name="hsv",
        file: str = None,
        custom_title: str = None,
        legend: bool = False,
        x_lim=(-10, 10),
        y_lim=(-10, 10),
        ax: Axes = None,
    ):
        if "x_y_location_metrics" not in self.metrics[algoname]:
            print(f"No x_y location metrics found for {algoname}")
            return

        if env_name not in self.metrics[algoname]["x_y_location_metrics"]:
            print(
                f"No x_y location metrics found for {algoname} in environment {env_name}"
            )
            return

        if ax is None:
            plt.figure(figsize=(8, 8))

        n_skills = len(
            self.metrics[algoname]["x_y_location_metrics"][env_name]
        )  # n_colors = n_skills
        # colors = plt.cm.get_cmap("tab20", max(n_skills, 1))
        cmap = plt.cm.get_cmap(cmap_name, n_skills)  # discrete version

        colors = cmap(np.arange(n_skills))

        for i, (skill, locations) in enumerate(
            self.metrics[algoname]["x_y_location_metrics"][env_name].items()
        ):
            x, y = zip(*locations)
            if ax is not None:
                ax.plot(x, y, label=f"skill {i}-th", color=colors[i])
            else:
                plt.plot(x, y, label=f"skill {i}-th", color=colors[i])
        if ax is None:
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
        else:
            ax.set_title(custom_title or f"{algoname} {env_name} Explore Metrics")
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            if legend:
                ax.legend()
        return ax

    def save_x_location(self, algoname: str, env_name: str, skill: Any, x: float):
        if env_name not in self.metrics[algoname]["x_location_metrics"]:
            self.metrics[algoname]["x_location_metrics"][env_name] = {}

        if skill not in self.metrics[algoname]["x_location_metrics"][env_name]:
            self.metrics[algoname]["x_location_metrics"][env_name][skill] = []

        self.metrics[algoname]["x_location_metrics"][env_name][skill].append(x)

    def plot_x_location_metrics(
        self,
        algoname: str,
        env_name: str,
        cmap_name="hsv",
        file: str = None,
        custom_title: str = None,
        legend: bool = False,
        x_lim=(-10, 10),
        ax: Axes = None,
    ):
        if "x_location_metrics" not in self.metrics[algoname]:
            print(f"No x location metrics found for {algoname}")
            return

        if env_name not in self.metrics[algoname]["x_location_metrics"]:
            print(
                f"No x location metrics found for {algoname} in environment {env_name}"
            )
            return

        if ax is None:
            plt.figure(figsize=(8, 8))

        n_skills = len(
            self.metrics[algoname]["x_location_metrics"][env_name]
        )  # n_colors = n_skills
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

            if ax is not None:
                ax.fill_between(
                    xs, y * 2 + density, y * 2 - density, color=colors[i], alpha=0.25
                )
            else:
                plt.fill_between(
                    xs, y * 2 + density, y * 2 - density, color=colors[i], alpha=0.25
                )

        if ax is None:
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
        else:
            ax.set_title(custom_title or f"{algoname} {env_name} X Location Metrics")
            # ax.set_xlabel('Time step')
            # ax.set_ylabel('X position')
            ax.set_yticks([])
            ax.set_xlim(x_lim)
            if legend:
                ax.legend()
        return ax

    def save_return(self, algoname: str, env_name: str, ret: float):
        """
        Use this for downstream tasks, where we only care about the return of the final policy.
        Use at the end of episode, you should run this for a certain number of episodes -> this will save and later can print the mean and std of the return.
        """
        if env_name not in self.metrics[algoname]["returns"]:
            self.metrics[algoname]["returns"][env_name] = []
        self.metrics[algoname]["returns"][env_name].append(ret)

    def set_returns(self, algoname: str, env_name: str, returns: list[float]):
        """
        Use this for training curves, where we care about the return of the policy during training.
        Use at the end of episode, you should run this for a certain number of episodes -> this will save and later can print the mean and std of the return.
        """
        self.metrics[algoname]["returns"][env_name] = returns

    @lru_cache(maxsize=128)
    def get_means_and_stds(
        self, algoname: str, env_name: str, window: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        returns = np.asarray(self.metrics[algoname]["returns"][env_name])

        means = np.zeros(len(returns))
        stds = np.zeros(len(returns))

        for i in range(len(returns)):
            start = max(0, i - window + 1)
            window_data = returns[start : i + 1]

            means[i] = np.mean(window_data)
            stds[i] = np.std(window_data)

        return means, stds

    def plot_running_returns(
        self,
        algonames: list[str],
        env_name: str,
        window: int = 100,
        x_lim=None,
        y_lim=None,
        file: str = None,
        custom_title: str = None,
        legend: bool = False,
        ax: Axes = None,
    ):
        if ax is None:
            plt.figure()

        for algoname in algonames:
            if "returns" not in self.metrics[algoname]:
                print(f"No return metrics found for {algoname}")
                continue

            if env_name not in self.metrics[algoname]["returns"]:
                print(
                    f"No return metrics found for {algoname} in environment {env_name}"
                )
                continue

            means, stds = self.get_means_and_stds(algoname, env_name, window)

            # Plot
            if ax is None:
                plt.plot(means, label=algoname)
                plt.fill_between(
                    range(len(means)),
                    means - stds,
                    means + stds,
                    alpha=0.3,
                )
            else:
                ax.plot(means, label=algoname)
                ax.fill_between(
                    range(len(means)),
                    means - stds,
                    means + stds,
                    alpha=0.3,
                )

        if ax is None:
            if x_lim is not None:
                plt.xlim(x_lim)
            if y_lim is not None:
                plt.ylim(y_lim)

            plt.xlabel("TimeStep")
            plt.ylabel("Return")
            if legend:
                plt.legend()
            plt.title(custom_title or f"Returns (window={window})")
            plt.show()
        else:
            if x_lim is not None:
                ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)

            ax.set_xlabel("TimeStep")
            ax.set_ylabel("Return")
            if legend:
                ax.legend()
            ax.set_title(custom_title or f"Returns (window={window})")

    def save_traj(
        self, algoname: str, env_name: str, skill: Any, traj_embed: np.ndarray
    ):
        """
        Save a trajectory for a specific skill in a specific environment.
        """

        if env_name not in self.metrics[algoname]["trajectory_embeddings"]:
            self.metrics[algoname]["trajectory_embeddings"][env_name] = {}
        if skill not in self.metrics[algoname]["trajectory_embeddings"][env_name]:
            self.metrics[algoname]["trajectory_embeddings"][env_name][skill] = []

        self.metrics[algoname]["trajectory_embeddings"][env_name][skill] = traj_embed

    def plot_traj_confusion_matrix(
        self,
        algoname: str,
        env_name: str,
        file: str = None,
        custom_title: str = None,
        ax: Axes = None,
    ) -> tuple[float, float, float]:
        """
        traj is list of observations
        encoder can turn each traj into an embedding space -> calc confusion matrix of the embeddings -> calculate the diff metric based on the confusion matrix
        return max, mean and min
        """
        # support either a single iterable of observations or multiple obs args

        # labels = list(self.metrics[algoname]["trajectory_embeddings"][env_name].keys())

        embeddings = [
            self.metrics[algoname]["trajectory_embeddings"][env_name][skill]
            for skill in self.metrics[algoname]["trajectory_embeddings"][env_name]
        ]

        if len(embeddings) == 0:
            print("No trajectory observations provided")
            return 0.0, 0.0, 0.0

        X = np.array(embeddings)

        np.random.shuffle(X)

        # cosine distance matrix: 0 = identical direction, 1 = orthogonal
        dist_matrix = cosine_distances(X)

        if ax is None:
            plt.figure(figsize=(7, 6))
            plt.imshow(dist_matrix, vmin=0, vmax=1, cmap="viridis")
            plt.colorbar(label="Cosine distance")
            plt.title(custom_title or "Trajectory Confusion Matrix")
            plt.tight_layout()
            plt.show()
        else:
            ax.imshow(dist_matrix, vmin=0, vmax=1, cmap="viridis")
            ax.colorbar(label="Cosine distance")
            ax.set_title(custom_title or "Trajectory Confusion Matrix")
            plt.colorbar(ax=ax, label="Cosine distance")

        # # return mean similarity as a simple metric
        # return float(np.mean(conf_mat))

        return (
            float(np.max(dist_matrix)),
            float(np.mean(dist_matrix)),
            float(np.min(dist_matrix)),
        )

    def plot_state_coverage_from_env(
        self, env_name: str, window_size: float = 0.5
    ) -> int:

        ret = {}

        for algoname in self.metrics:
            use_x_y = True
            try:
                data = self.metrics[algoname]["x_y_location_metrics"][env_name]
            except KeyError:
                try:
                    data = self.metrics[algoname]["x_location_metrics"][env_name]
                    use_x_y = False
                except KeyError:
                    continue

            counted_dict = {}
            ret[algoname] = []

            for i in range(
                len(data[list(data.keys())[0]])
            ):  # for each time step, we assume each skill has the same number of data points
                for skill in data:
                    if use_x_y:
                        x_y = data[skill][i]
                        x = x_y[0]
                        y = x_y[1]
                        x_windowed = int(x // window_size) * window_size
                        y_windowed = int(y // window_size) * window_size
                        counted_dict[(x_windowed, y_windowed)] = (
                            counted_dict.get((x_windowed, y_windowed), 0) + 1
                        )

                    else:
                        x = data[skill][i]
                        x_windowed = int(x // window_size) * window_size
                        counted_dict[x_windowed] = counted_dict.get(x_windowed, 0) + 1

                ret[algoname].append(len(counted_dict))

            plt.plot(ret[algoname], label=algoname)

        plt.title(f"State Coverage in {env_name}")
        plt.xlabel("Time step")
        plt.ylabel("Unique states visited (windowed)")
        plt.legend()
        plt.show()

    def plot_percentage_state_coverage(
        self,
        algonames: list[str],
        env_name: str,
        window_size: float = 0.5,
        lower_bound: int = 1,
        upper_bound=None,
    ):
        ret: dict[str, list[int]] = {}

        for algoname in algonames:
            use_x_y = True
            try:
                data = self.metrics[algoname]["x_y_location_metrics"][env_name]
            except KeyError:
                try:
                    data = self.metrics[algoname]["x_location_metrics"][env_name]
                    use_x_y = False
                except KeyError:
                    continue

            counted_dict = {}
            ret[algoname] = []

            for i in range(
                len(data[list(data.keys())[0]])
            ):  # for each time step, we assume each skill has the same number of data points
                for skill in list(data.keys())[:16]:
                    if use_x_y:
                        x_y = data[skill][i]
                        x = x_y[0]
                        y = x_y[1]
                        x_windowed = int(x // window_size) * window_size
                        y_windowed = int(y // window_size) * window_size
                        counted_dict[(x_windowed, y_windowed)] = (
                            counted_dict.get((x_windowed, y_windowed), 0) + 1
                        )

                    else:
                        x = data[skill][i]
                        x_windowed = int(x // window_size) * window_size
                        counted_dict[x_windowed] = counted_dict.get(x_windowed, 0) + 1

                ret[algoname].append(
                    len(
                        [
                            loc
                            for loc, count in counted_dict.items()
                            if count >= lower_bound
                            and (upper_bound is None or count <= upper_bound)
                        ]
                    )
                    / len(counted_dict)
                    if len(counted_dict) > 0
                    else 0
                )

            plt.plot(ret[algoname], label=algoname)

        plt.title(f"State Coverage in {env_name}")
        plt.xlabel("Time step")
        # plt.ylim((0, 1))
        plt.ylabel("Unique states visited (windowed)")
        plt.legend()
        plt.show()

        return ret
