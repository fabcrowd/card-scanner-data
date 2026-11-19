"""The hosts a new-card picture address may start with. One list, read by the nightly build (new_sets.py) and by the
signer (scripts/sign_release.py). It holds constants only and imports nothing, so the sign job can import it with just
the standard library and `cryptography` installed."""
TCGDEX_IMAGE_HOST = "https://assets.tcgdex.net/"
POKEMONTCG_IMAGE_HOST = "https://images.pokemontcg.io/"
IMAGE_HOSTS = (TCGDEX_IMAGE_HOST, POKEMONTCG_IMAGE_HOST)
