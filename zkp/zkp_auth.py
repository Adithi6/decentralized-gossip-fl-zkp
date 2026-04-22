import json
import subprocess
import os


def generate_zkp_proof(
    client_id: str,
    secret: str,
    input_dir: str,
    proof_dir: str,
    build_dir: str,
    zkey_path: str,
):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(proof_dir, exist_ok=True)

    input_file = os.path.join(input_dir, f"{client_id}_input.json")
    witness_file = os.path.join(proof_dir, f"{client_id}_witness.wtns")
    proof_file = os.path.join(proof_dir, f"{client_id}_proof.json")
    public_file = os.path.join(proof_dir, f"{client_id}_public.json")

    # ⚠️ TEMP FIX: we assume commitment is already correct
    # Later we can compute dynamically
    input_data = {
        "secret": secret,
        "commitment": "4267533774488295900887461483015112262021273608761099826938271132511348470966"
    }

    with open(input_file, "w") as f:
        json.dump(input_data, f)

    # Generate witness
    subprocess.run([
        "node",
        os.path.join(build_dir, "client_auth_js", "generate_witness.js"),
        os.path.join(build_dir, "client_auth_js", "client_auth.wasm"),
        input_file,
        witness_file
    ], check=True)

    # Generate proof
    subprocess.run([
        "snarkjs", "groth16", "prove",
        zkey_path,
        witness_file,
        proof_file,
        public_file
    ], check=True)

    return proof_file, public_file
