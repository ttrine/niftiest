from abc import ABC, abstractmethod
import logging
import pickle as pkl
import warnings

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from utils import *
from preprocess import DATA_DIR, preprocess

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommenderSystem(ABC):
    def __init__(self, user_graph):
        self.graph = self._clean(user_graph)
        self._standardize(self.graph)

        # To check if we can make recs to the active user
        self.valid_users = set(self.graph)

        self.model = None
        self.scores = None
        self.k_max = False

        # Give each graph node a place to record eval metrics
        for _, user_data in self.graph.nodes(data=True):
            user_data['metrics'] = UserMetrics()

    def _clean(self, graph):
        """ Filter out users without enough associated information to make recommendations. """

        logger.info(f"Input data has {len(graph.nodes)} users and {len(graph.edges)} island visits.")

        # Remove edges with weight 0
        e = [(node1, node2) for node1, node2, t in graph.edges(data='totalTime') if t == 0]
        graph.remove_edges_from(e)

        # Remove users who haven't visited at least two islands
        graph = nx.DiGraph(graph.subgraph([node for node in graph if len(graph.edges(node)) >= 2]))

        logger.info(f"Data contains {len(graph.nodes)} users and {len(graph.edges)} island visits after cleaning.")

        return graph

    def _standardize(self, graph):
        """ Shift and scale the data to make it suitable for training a recommendation system. """

        # Convert visitation lengths to log scale
        for _, _, data in graph.edges(data=True):
            data['logTime'] = np.log(data['totalTime'])

        # Standardize per-user distributions by subtracting the mean, dividing by standard deviation
        for user, user_data in graph.nodes(data=True):
            interactions = [lt for _, _, lt in graph.edges(user, data='logTime')]

            user_data['mean'] = np.mean(interactions)
            user_data['std'] = np.std(interactions)

            for _, island, data in graph.edges(user, data=True):
                if len(interactions) == 1:
                    # Avoid nans resulting from division by 0
                    data['standardized'] = 0.
                    continue
                data['standardized'] = (data['logTime'] - user_data['mean']) / user_data['std']

    @abstractmethod
    def fit(self):
        """ Train the recommender system on the input data.

        Implementations should train a model and assign it to self.model. """
        pass

    @abstractmethod
    def score(self):
        """ Produce a score for every pair of distinct nodes in the graph.

        Implementations should construct a dict of dicts of scores and assign it to self.scores. """
        pass

    def compute_user_metrics(self, k_max=300, thresh=-10000.):
        """ Calculate per-user performance metrics against the test set. """
        assert self.scores, "Please generate scores first"

        # TODO Optimize this, there's a lot of redundant computation
        for k in range(1, k_max):
            for user, user_data in self.graph.nodes(data=True):
                # Get top k recommended islands for user
                top_k = sorted(self.scores[user].items(), reverse=True, key=lambda x: x[1])[:k]
                top_k_islands, top_k_scores = list(zip(*top_k))

                # Get all 'relevant' islands for user
                ground_truth = {island for island in self.graph[user]
                                if self.graph[user][island]['standardized'] > thresh}

                # Record performance metrics
                metrics = user_data['metrics']
                numer = len(set(top_k_islands).intersection(ground_truth))
                metrics.precision[k] = numer / k
                metrics.recall[k] = numer / len(ground_truth) if len(ground_truth) > 0 else 1.
                metrics.avg_precision[k] = apk(ground_truth, top_k_islands, k)

        self.k_max = k_max

    def get_metrics_for_user(self, user):
        """ Retrieve the metrics for a given user. """
        assert user in self.valid_users, "Invalid user. Either the ID is incorrect or there's insufficient data."
        assert self.k_max, "Please call compute_user_metrics first with the desired hyperparameters"

        avg_precision_list = list(self.graph.nodes[user]['metrics'].avg_precision.values())
        recall_list = list(self.graph.nodes[user]['metrics'].recall.values())

        return avg_precision_list, recall_list

    def get_average_metrics(self):
        """ Average the per-user metrics into global (averaged across user) performance measures. """
        assert self.k_max, "Please call compute_user_metrics first with the desired hyperparameters"

        m_av_prec = list()
        m_rec = list()
        for k in range(1, self.k_max):
            m_av_prec.append(np.mean([user_data['metrics'].avg_precision[k]
                                      for user, user_data in self.graph.nodes(data=True)]))
            m_rec.append(np.mean([user_data['metrics'].recall[k] for user, user_data in self.graph.nodes(data=True)]))

        return m_av_prec, m_rec


class SimrankBaseline(RecommenderSystem):
    """ Score nodes based on unweighted, undirected graph topology alone.

    This does not require learning and should be relatively easy to beat. """

    def __init__(self, user_graph):
        super().__init__(user_graph)

    def fit(self):
        """ Do nothing, because Simrank scoring is not based on a statistical model."""
        pass

    def score(self):
        """ Produce Simrank scores for all node pairs. """
        scores = nx.simrank_similarity(self.graph)

        # Remove self-edges, which always have a score of 1
        for node in self.graph:
            if node in scores[node]:
                del scores[node][node]

        self.scores = scores


def maybe_create_recsys():
    """ Check for a serialized version of the SimrankBaseline class, load it if it's there, create it if not.

    In a production implementation, this will be parameterized by choice of model. """
    try:
        with open(f'{DATA_DIR}/recsys.pkl', 'rb') as f:
            recsys = pkl.load(f)
    except Exception:
        logger.info("Serialized recsys not found (or other error encountered), attempting to create it")
        try:
            graph = nx.read_gpickle(f"{DATA_DIR}/graph.pkl")
        except FileNotFoundError:
            logger.info("Preprocessed data not found, running preprocessing pipeline")
            graph = preprocess(serialize=True)

        recsys = SimrankBaseline(graph)
        recsys.score()
        recsys.compute_user_metrics()

        with open(f'{DATA_DIR}/recsys.pkl', 'wb') as f:
            pkl.dump(recsys, f)

    return recsys


if __name__ == '__main__':
    recsys = maybe_create_recsys()
    logger.info(f"Saving average PR curve plot to {DATA_DIR}/plots/average.jpg")

    m_av_prec, m_rec = recsys.get_average_metrics()

    fig, ax = plt.subplots()
    ax.plot(m_rec, m_av_prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision as a Function of Recall Averaged Across Users")
    plt.savefig('data/plots/average_performance_across_users.jpg')
    plt.close(fig)
