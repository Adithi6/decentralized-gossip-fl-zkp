import logging
from torch.utils.data import DataLoader

import numpy as np

from client.fl_client import FederatedClient
from utils.weights import bytes_to_weight_arrays, apply_weight_arrays


class GossipNode:
    """
    A GossipNode = FederatedClient + gossip inbox.
    """

    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        device: str,
        use_hash: bool,
        hash_algorithm: str,
        weight_dtype: str,
        learning_rate: float,
        crypto_scheme: str,
        model_name: str,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
        hidden_dim: int,
    ):
        self.client = FederatedClient(
            client_id=client_id,
            dataloader=dataloader,
            device=device,
            use_hash=use_hash,
            hash_algorithm=hash_algorithm,
            weight_dtype=weight_dtype,
            learning_rate=learning_rate,
            crypto_scheme=crypto_scheme,
            model_name=model_name,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        )

        self.own_submission: dict | None = None
        self.inbox: list[dict] = []

        self.client_id = client_id
        self.pk = self.client.pk

        logging.info(
            f"[{self.client_id}] gossip node initialized | "
            f"use_hash={use_hash} hash_algorithm={hash_algorithm} "
            f"weight_dtype={weight_dtype} learning_rate={learning_rate} "
            f"scheme={crypto_scheme} model={model_name}"
        )

    def local_train(self, global_weight_arrays: list | None, epochs: int = 1):
        self.client.local_train(global_weight_arrays, epochs)

    def sign_update(self) -> dict:
        self.own_submission = self.client.sign_update()
        self.inbox = []
        logging.info(f"[{self.client_id}] own submission stored and inbox reset")
        return self.own_submission

    def receive_gossip(self, message: dict):
        already_have = any(
            m["payload"] == message["payload"] for m in self.inbox
        )

        if not already_have:
            self.inbox.append(message)
            logging.info(
                f"[{self.client_id}] received gossip from {message['client_id']} "
                f"| inbox_size={len(self.inbox)}"
            )
        else:
            logging.warning(
                f"[{self.client_id}] duplicate gossip ignored from {message['client_id']}"
            )

    def get_all_submissions(self) -> list[dict]:
        all_subs = []
        if self.own_submission:
            all_subs.append(self.own_submission)
        all_subs.extend(self.inbox)
        return all_subs

    def aggregate_local_updates(self, submissions: list[dict], template_model):
        if not submissions:
            logging.warning(f"[{self.client_id}] no submissions available for aggregation")
            return

        logging.info(f"[{self.client_id}] aggregating {len(submissions)} submission(s)")

        # infer dtype from the client's configured weight dtype
        dtype_name = self.client.weight_dtype

        weight_sets = []
        for sub in submissions:
            arrays = bytes_to_weight_arrays(
                sub["update_bytes"],
                template_model,
                dtype_name=dtype_name,
            )
            weight_sets.append(arrays)

        averaged = [
            np.mean([weights[i] for weights in weight_sets], axis=0)
            for i in range(len(weight_sets[0]))
        ]

        apply_weight_arrays(self.client.model, averaged)
        logging.info(f"[{self.client_id}] local aggregation completed")