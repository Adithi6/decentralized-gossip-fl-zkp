import logging
import yaml
import time
import random
import json
import base64
import torch

from data.loader import make_client_loaders
from gossip.node import GossipNode
from gossip.protocol import GossipProtocol
from utils.weights import model_to_weight_arrays


REGISTRY_FILE = "client_registry.json"


def setup_logging(config):
    logging.basicConfig(
        level=getattr(logging, config["logging"]["log_level"]),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(config["logging"]["log_file"]),
            logging.StreamHandler()
        ]
    )


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_public_keys_to_json(nodes, path):
    registry = {
        node.client_id: base64.b64encode(node.pk).decode("utf-8")
        for node in nodes
    }
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    logging.info(f"Public key registry saved to {path}")


def load_public_keys_from_json(path):
    with open(path, "r") as f:
        registry = json.load(f)
    return {
        cid: base64.b64decode(pk.encode("utf-8"))
        for cid, pk in registry.items()
    }


def choose_aggregator_node(nodes):
    counts = [(node, len(node.get_all_submissions())) for node in nodes]

    for node, count in counts:
        logging.info(f"{node.client_id} submissions = {count}")

    max_count = max(c for _, c in counts)
    candidates = [n for n, c in counts if c == max_count]

    aggregator = random.choice(candidates)
    logging.info(f"Selected aggregator: {aggregator.client_id}")
    return aggregator


def main():
    config = load_config()
    setup_logging(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # -------- CONFIG --------
    N_CLIENTS = config["experiment"]["n_clients"]
    N_ROUNDS = config["experiment"]["n_rounds"]
    LOCAL_EPOCHS = config["experiment"]["local_epochs"]

    GOSSIP_FANOUT = config["gossip"]["fanout"]
    GOSSIP_MAX_HOPS = config["gossip"]["max_hops"]

    USE_HASH = config["security"]["use_hash"]
    HASH_ALGO = config["security"]["hash_algorithm"]

    CRYPTO_SCHEME = config["crypto"]["scheme"]

    LEARNING_RATE = config["training"]["learning_rate"]

    MODEL = config["model"]
    DATA = config["data"]
    WEIGHTS = config["weights"]

    # -------- OPTIONAL ZKP --------
    ZKP = config.get("zkp", {})
    ZKP_ENABLED = ZKP.get("enabled", True)
    ZKP_SECRET = ZKP.get("secret", "12345")
    ZKP_INPUT_DIR = ZKP.get("input_dir", "zkp/inputs")
    ZKP_PROOF_DIR = ZKP.get("proof_dir", "zkp/proofs")
    ZKP_BUILD_DIR = ZKP.get("build_dir", "zkp/build")
    ZKP_ZKEY_PATH = ZKP.get("zkey_path", "zkp/keys/client_auth_final.zkey")

    # -------- DATA --------
    client_loaders, _ = make_client_loaders(
        n_clients=N_CLIENTS,
        batch_size=DATA["batch_size"],
        alpha=DATA["alpha"],
        dataset_name=DATA["dataset_name"],
        partition_by=DATA["partition_by"],
        min_partition_size=DATA["min_partition_size"],
        self_balancing=DATA["self_balancing"],
        seed=DATA["seed"],
        test_batch_size=DATA["test_batch_size"],
        normalize_mean=DATA["normalize_mean"],
        normalize_std=DATA["normalize_std"],
    )

    # -------- NODES --------
    nodes = []
    for i in range(N_CLIENTS):
        node = GossipNode(
            client_id=f"client_{i}",
            dataloader=client_loaders[i],
            device=device,
            use_hash=USE_HASH,
            hash_algorithm=HASH_ALGO,
            weight_dtype=WEIGHTS["dtype"],
            learning_rate=LEARNING_RATE,
            crypto_scheme=CRYPTO_SCHEME,
            model_name=MODEL["name"],
            input_channels=MODEL["input_channels"],
            num_classes=MODEL["num_classes"],
            input_height=MODEL["input_height"],
            input_width=MODEL["input_width"],
            conv1_channels=MODEL["conv1_channels"],
            conv2_channels=MODEL["conv2_channels"],
            hidden_dim=MODEL["hidden_dim"],
            zkp_enabled=ZKP_ENABLED,
            zkp_secret=ZKP_SECRET,
            zkp_input_dir=ZKP_INPUT_DIR,
            zkp_proof_dir=ZKP_PROOF_DIR,
            zkp_build_dir=ZKP_BUILD_DIR,
            zkp_zkey_path=ZKP_ZKEY_PATH,
        )
        nodes.append(node)

    save_public_keys_to_json(nodes, REGISTRY_FILE)
    all_pub_keys = load_public_keys_from_json(REGISTRY_FILE)

    # -------- GOSSIP --------
    gossip = GossipProtocol(
        fanout=GOSSIP_FANOUT,
        max_hops=GOSSIP_MAX_HOPS,
        all_pub_keys=all_pub_keys,
        crypto_scheme=CRYPTO_SCHEME,
    )

    # -------- INIT GLOBAL MODEL --------
    initializer = random.choice(nodes)
    logging.info(f"Selected initializer node: {initializer.client_id}")

    global_weights = model_to_weight_arrays(initializer.client.model)

    # Broadcast initial common model to all nodes
    for node in nodes:
        node.local_train(global_weights, epochs=0)

    # -------- TRAINING --------
    start_time = time.time()

    for r in range(1, N_ROUNDS + 1):
        logging.info(f"Round {r}/{N_ROUNDS} started")

        # all nodes train from the same current global model
        for node in nodes:
            node.local_train(global_weights, epochs=LOCAL_EPOCHS)

        # each node signs its local update
        for node in nodes:
            node.sign_update()

        # gossip propagation + verification
        gossip.run_round(nodes)
        gossip.print_gossip_summary()

        # choose aggregator
        aggregator = choose_aggregator_node(nodes)
        subs = aggregator.get_all_submissions()

        if subs:
            # aggregator computes updated model
            aggregator.aggregate_local_updates(subs, aggregator.client.model)

            # aggregator model becomes next round's global model
            global_weights = model_to_weight_arrays(aggregator.client.model)

            logging.info(
                f"[{aggregator.client_id}] broadcasting updated global model to all nodes"
            )

            # broadcast updated model to all nodes
            for node in nodes:
                node.local_train(global_weights, epochs=0)

        logging.info(f"Round {r}/{N_ROUNDS} completed")

    end_time = time.time()
    logging.info(f"Total execution time = {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()