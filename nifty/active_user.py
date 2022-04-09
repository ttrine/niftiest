import logging
import sys

from matplotlib import pyplot as plt

from recommend import maybe_create_recsys
from preprocess import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Please supply the ID of the active user"
    active_user = sys.argv[1]
    recsys = maybe_create_recsys()

    logger.info(f"Attempting to retrieve metrics for user {active_user}")
    avg_precision_list, recall_list = recsys.get_metrics_for_user(active_user)

    out_dir = f"{DATA_DIR}/plots/{active_user}_performance.jpg"
    logger.info(f"Saving user PR curve to {out_dir}")
    fig, ax = plt.subplots()
    ax.plot(recall_list, avg_precision_list)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"APR Curve for User {active_user}")
    plt.savefig(out_dir)
    plt.close(fig)
