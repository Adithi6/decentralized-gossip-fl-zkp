import logging
import time


def _get_scheme(scheme_name: str):
    """
    Dynamically load PQC scheme.
    Currently supports Dilithium variants.
    """
    scheme_name = scheme_name.lower()

    if scheme_name == "dilithium2":
        from dilithium_py.dilithium import Dilithium2
        return Dilithium2

    if scheme_name == "dilithium3":
        from dilithium_py.dilithium import Dilithium3
        return Dilithium3

    if scheme_name == "dilithium5":
        from dilithium_py.dilithium import Dilithium5
        return Dilithium5

    raise ValueError(f"Unsupported crypto scheme: {scheme_name}")


def keygen(scheme_name: str) -> tuple[bytes, bytes, float]:
    Scheme = _get_scheme(scheme_name)

    t0 = time.perf_counter()
    pk, sk = Scheme.keygen()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.debug(f"{scheme_name} keygen completed in {elapsed_ms:.2f} ms")
    return pk, sk, elapsed_ms


def sign(sk: bytes, message: bytes, scheme_name: str) -> tuple[bytes, float]:
    Scheme = _get_scheme(scheme_name)

    t0 = time.perf_counter()
    signature = Scheme.sign(sk, message)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.debug(
        f"{scheme_name} signing completed in {elapsed_ms:.2f} ms "
        f"(input={len(message)} bytes)"
    )
    return signature, elapsed_ms


def verify(pk: bytes, message: bytes, signature: bytes, scheme_name: str) -> tuple[bool, float]:
    Scheme = _get_scheme(scheme_name)

    t0 = time.perf_counter()
    is_valid = Scheme.verify(pk, message, signature)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logging.debug(
        f"{scheme_name} verify completed in {elapsed_ms:.2f} ms | valid={is_valid}"
    )
    return is_valid, elapsed_ms