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

        # 🔥 ZKP CONFIG (NEW)
        zkp_enabled=True,
        zkp_secret="12345",
        zkp_input_dir="zkp/inputs",
        zkp_proof_dir="zkp/proofs",
        zkp_build_dir="zkp/build",
        zkp_zkey_path="zkp/keys/client_auth_final.zkey",
    )
    nodes.append(node)