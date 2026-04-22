import logging
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.cnn import SmallCNN
from crypto import dilithium_utils
from utils.weights import apply_weight_arrays, weights_to_bytes
from zkp.zkp_auth import generate_zkp_proof


def build_model(
    model_name: str,
    device: str,
    input_channels: int,
    num_classes: int,
    input_height: int,
    input_width: int,
    conv1_channels: int,
    conv2_channels: int,
    hidden_dim: int,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "smallcnn":
        return SmallCNN(
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        ).to(device)

    raise ValueError(f"Unsupported model: {model_name}")


class FederatedClient:
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
        zkp_enabled: bool,
        zkp_secret: str,
        zkp_input_dir: str,
        zkp_proof_dir: str,
        zkp_build_dir: str,
        zkp_zkey_path: str,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.use_hash = use_hash
        self.hash_algorithm = hash_algorithm
        self.weight_dtype = weight_dtype
        self.learning_rate = learning_rate
        self.crypto_scheme = crypto_scheme
        self.model_name = model_name

        self.zkp_enabled = zkp_enabled
        self.zkp_secret = zkp_secret
        self.zkp_input_dir = zkp_input_dir
        self.zkp_proof_dir = zkp_proof_dir
        self.zkp_build_dir = zkp_build_dir
        self.zkp_zkey_path = zkp_zkey_path

        self.model = build_model(
            model_name=self.model_name,
            device=self.device,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
            hidden_dim=hidden_dim,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.pk, self.sk, keygen_ms = dilithium_utils.keygen(self.crypto_scheme)
        self.sign_ms: float | None = None

        logging.info(
            f"[{client_id}] keygen: {keygen_ms:.2f} ms "
            f"(pk={len(self.pk)}B sk={len(self.sk)}B) | scheme={self.crypto_scheme}"
        )
        logging.info(
            f"[{client_id}] training config | "
            f"model={self.model_name} learning_rate={self.learning_rate} "
            f"use_hash={self.use_hash} hash_algorithm={self.hash_algorithm} "
            f"weight_dtype={self.weight_dtype} zkp_enabled={self.zkp_enabled}"
        )

    def local_train(self, global_weight_arrays=None, epochs=1):
        if global_weight_arrays is not None:
            apply_weight_arrays(self.model, global_weight_arrays)

        if epochs == 0:
            return

        self.model.train()
        total_loss = 0.0

        for _ in range(epochs):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(x)
                loss = self.criterion(logits, y)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx == 0:
                    pred = torch.argmax(logits, dim=1)
                    logging.info(
                        f"[{self.client_id}] logits sample: {logits[0].detach().cpu().numpy()} | "
                        f"pred={pred[0].item()} | actual={y[0].item()}"
                    )

        total_batches = len(self.dataloader) * epochs
        logging.info(
            f"[{self.client_id}] trained | loss: {total_loss / total_batches:.4f}"
        )

    def _hash_payload(self, update_bytes: bytes) -> bytes:
        algo = self.hash_algorithm.lower()

        if algo == "sha256":
            return hashlib.sha256(update_bytes).digest()

        if algo == "sha512":
            return hashlib.sha512(update_bytes).digest()

        raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

    def sign_update(self) -> dict:
        update_bytes = weights_to_bytes(self.model, self.weight_dtype)

        if self.use_hash:
            payload = self._hash_payload(update_bytes)
            mode = "HASHED"
            hash_algorithm = self.hash_algorithm
        else:
            payload = update_bytes
            mode = "RAW"
            hash_algorithm = ""

        signature, sign_ms = dilithium_utils.sign(
            self.sk,
            payload,
            self.crypto_scheme
        )
        self.sign_ms = float(sign_ms)

        zkp_proof = None
        zkp_public = None

        if self.zkp_enabled:
            zkp_proof, zkp_public = generate_zkp_proof(
                client_id=self.client_id,
                secret=self.zkp_secret,
                input_dir=self.zkp_input_dir,
                proof_dir=self.zkp_proof_dir,
                build_dir=self.zkp_build_dir,
                zkey_path=self.zkp_zkey_path,
            )
            logging.info(f"[{self.client_id}] ZKP proof generated successfully")

        logging.info(
            f"[{self.client_id}] signed ({mode}) | {sign_ms:.3f} ms "
            f"scheme={self.crypto_scheme} "
            f"input={len(payload)} B update={len(update_bytes)/1024:.1f} KB "
            f"sig={len(signature)} B"
        )

        return {
            "client_id": self.client_id,
            "update_bytes": update_bytes,
            "payload": payload,
            "signature": signature,
            "sign_ms": float(sign_ms),
            "is_hashed": self.use_hash,
            "hash_algorithm": hash_algorithm,
            "zkp_proof": zkp_proof,
            "zkp_public": zkp_public,
        }