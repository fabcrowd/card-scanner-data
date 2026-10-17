"""ECDSA P-256 / SHA-256 signing. Signatures are raw r||s (64 bytes, base64), what WebCrypto verify expects."""
import base64
import binascii

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature


def generate_keypair():
    """(PKCS8 PEM private key bytes, base64 SPKI DER public key) for a new P-256 key."""
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    spki = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return pem, base64.b64encode(spki).decode()


def sign_bytes(data, pem):
    """Base64 raw r||s signature of data. Raises ValueError when there is no usable key."""
    if not pem:
        raise ValueError("signing key is empty")
    try:
        key = serialization.load_pem_private_key(pem if isinstance(pem, bytes) else pem.encode(), password=None)
    except (ValueError, TypeError) as err:
        raise ValueError(f"signing key is not a PKCS8 PEM: {err}") from err
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ValueError("signing key must be ECDSA P-256")
    r, s = decode_dss_signature(key.sign(data, ec.ECDSA(hashes.SHA256())))
    return base64.b64encode(r.to_bytes(32, "big") + s.to_bytes(32, "big")).decode()


def verify_signature(data, sig_b64, spki_b64):
    try:
        raw = base64.b64decode(sig_b64, validate=True)
        if len(raw) != 64:
            return False
        pub = serialization.load_der_public_key(base64.b64decode(spki_b64))
        pub.verify(encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")), data, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, binascii.Error, ValueError, TypeError):
        return False
