"""Make SearxNG trust local TLS-intercepting roots (e.g. Kaspersky).

Runs inside the SearxNG container at startup (wired up via the compose
`entrypoint`). Some machines run antivirus/proxies (Kaspersky, Zscaler, ...)
that intercept HTTPS with their own root CA. The container doesn't trust that
root, so affected search engines fail with "HTTP connection error".

Tricky part: interceptors like Kaspersky often re-sign only SOME domains, so a
single probe (e.g. google) can look fine while wikipedia/brave/startpage are
intercepted. We therefore probe every engine host SearxNG uses; for any whose
TLS chain does NOT verify, we capture the chain the interceptor presents *to
this Python process* and append its certs to certifi's CA bundle (which
SearxNG's httpx uses). Idempotent and best-effort: if verification already
works, or anything goes wrong, SearxNG still starts.

Requires Python 3.13+ for ssl.SSLSocket.get_unverified_chain() (image ships 3.14).
"""
import socket
import ssl

# Hosts SearxNG's default engines actually contact.
PROBE_HOSTS = [
    "en.wikipedia.org",
    "duckduckgo.com",
    "html.duckduckgo.com",
    "www.google.com",
    "search.brave.com",
    "www.startpage.com",
]
PORT = 443
TIMEOUT = 8


def _verifies(host: str, cafile: str) -> bool:
    """True if a normal verified TLS handshake to `host` succeeds."""
    try:
        ctx = ssl.create_default_context(cafile=cafile)
        with socket.create_connection((host, PORT), timeout=TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                return True
    except Exception:
        return False


def _capture_chain_pems(host: str) -> list:
    """Return the interceptor's cert chain for `host` as a list of PEM strings."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, PORT), timeout=TIMEOUT) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as tls:
            chain = tls.get_unverified_chain() or []
    pems = []
    for cert in chain:
        try:
            der = bytes(cert) if isinstance(cert, (bytes, bytearray)) else cert.public_bytes(ssl.Encoding.DER)
            pems.append(ssl.DER_cert_to_PEM_cert(der))
        except Exception:
            continue
    return pems


def main() -> None:
    try:
        import certifi
    except Exception:
        print("[trust-ca] certifi not found; skipping.")
        return
    cafile = certifi.where()
    try:
        with open(cafile, encoding="ascii", errors="ignore") as fh:
            existing = fh.read()
    except Exception:
        existing = ""

    collected = []
    seen = set()
    for host in PROBE_HOSTS:
        try:
            if _verifies(host, cafile):
                continue  # not intercepted (or already trusted)
            for pem in _capture_chain_pems(host):
                key = pem.strip()
                if key and key not in seen and key not in existing:
                    seen.add(key)
                    collected.append(pem)
        except Exception as exc:
            print(f"[trust-ca] probe {host} failed: {exc!r}")

    if not collected:
        print("[trust-ca] nothing to add (TLS already trusted, or nothing captured).")
        return

    try:
        with open(cafile, "a", encoding="ascii") as fh:
            fh.write("\n" + "".join(collected))
    except Exception as exc:
        print(f"[trust-ca] could not append to certifi bundle: {exc!r}")
        return

    ok = sum(1 for h in PROBE_HOSTS if _verifies(h, cafile))
    print(f"[trust-ca] added {len(collected)} interception cert(s); {ok}/{len(PROBE_HOSTS)} probe hosts verify now.")


if __name__ == "__main__":
    main()
