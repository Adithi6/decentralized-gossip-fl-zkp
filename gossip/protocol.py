import logging
import random
import hashlib
import subprocess

from gossip.node import GossipNode
from crypto import dilithium_utils


# -------- FIXED PATHS (VERY IMPORTANT) --------
SNARKJS_PATH = r"C:\Users\ADITHI\AppData\Roaming\npm\snarkjs.cmd"
VERIFICATION_KEY_PATH = r"zkp/keys/verification_key.json"


class GossipProtocol:
    def __init__(
        self,
        fanout: int,
        max_hops: int,
        all_pub_keys: dict[str, bytes],
        crypto_scheme: str,
    ):
        self.fanout = fanout
        self.max_hops = max_hops
        self.all_pub_keys = all_pub_keys
        self.crypto_scheme = crypto_scheme

        self._seen = set()
        self.gossip_timings = []

    def reset_round(self):
        self._seen.clear()
        self.gossip_timings.clear()
        logging.info("Gossip round state reset")

    # -------- HASH CHECK --------
    def _compute_expected_payload(self, message: dict) -> bytes:
        if not message["is_hashed"]:
            return message["update_bytes"]

        algo = message["hash_algorithm"].lower()

        if algo == "sha256":
            return hashlib.sha256(message["update_bytes"]).digest()
        elif algo == "sha512":
            return hashlib.sha512(message["update_bytes"]).digest()

        raise ValueError("Unsupported hash algorithm")

    # -------- VERIFY --------
    def _verify_before_forward(self, message: dict):
        pk = self.all_pub_keys.get(message["client_id"])

        if pk is None:
            logging.error(f"Missing public key for {message['client_id']}")
            return False, 0.0

        expected_payload = self._compute_expected_payload(message)

        if expected_payload != message["payload"]:
            logging.warning("Payload mismatch")
            return False, 0.0

        # -------- DILITHIUM VERIFY --------
        is_valid, verify_ms = dilithium_utils.verify(
            pk,
            message["payload"],
            message["signature"],
            self.crypto_scheme,
        )

        if not is_valid:
            logging.warning(f"Signature failed for {message['client_id']}")
            return False, verify_ms

        # -------- ZKP VERIFY --------
        if message.get("zkp_proof"):
            proof_file = message["zkp_proof"]
            public_file = message["zkp_public"]

            try:
                result = subprocess.run(
                    [
                        SNARKJS_PATH,
                        "groth16",
                        "verify",
                        VERIFICATION_KEY_PATH,
                        public_file,
                        proof_file,
                    ],
                    capture_output=True,
                    text=True,
                )

                if "OK!" not in result.stdout:
                    logging.warning(f"ZKP failed for {message['client_id']}")
                    return False, verify_ms

                logging.info(f"[ZKP] verified for {message['client_id']}")

            except Exception as e:
                logging.error(f"ZKP error: {e}")
                return False, verify_ms

        return True, verify_ms

    # -------- GOSSIP --------
    def spread(self, origin_node, all_nodes, message, hop=0):

        message_id = (message["client_id"], message["payload"])
        state_id = (message_id, origin_node.client_id)

        if state_id in self._seen:
            return

        if hop >= self.max_hops:
            return

        self._seen.add(state_id)

        peers = [n for n in all_nodes if n.client_id != origin_node.client_id]
        targets = random.sample(peers, min(self.fanout, len(peers)))

        for target in targets:

            is_valid, verify_ms = self._verify_before_forward(message)

            self.gossip_timings.append({
                "from": origin_node.client_id,
                "to": target.client_id,
                "hop": hop + 1,
                "verify_ms": round(verify_ms, 3),
                "accepted": is_valid,
            })

            logging.info(
                f"[gossip] {origin_node.client_id} -> {target.client_id} "
                f"[{'OK' if is_valid else 'REJECTED'}]"
            )

            if is_valid:
                target.receive_gossip(message)
                self.spread(target, all_nodes, message, hop + 1)

    # -------- RUN ROUND --------
    def run_round(self, nodes):

        self.reset_round()

        for node in nodes:
            if node.own_submission is None:
                raise RuntimeError("Call sign_update() first")

            self.spread(node, nodes, node.own_submission)

    # -------- SUMMARY --------
    def print_gossip_summary(self):
        if not self.gossip_timings:
            return

        logging.info("---- Gossip Summary ----")

        for t in self.gossip_timings:
            logging.info(
                f"{t['from']} -> {t['to']} | hop={t['hop']} | "
                f"verify={t['verify_ms']} | accepted={t['accepted']}"
            )