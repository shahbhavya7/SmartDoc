"""Authentication owned by FastAPI: bcrypt passwords, JWTs, Google OAuth.

The API issues and verifies its own tokens. No identity decision is delegated
to the frontend, so a client can present a token and nothing else -- there is no
request field anywhere in the system that names a user.

The one rule everything else depends on
---------------------------------------
``user_id`` comes from :func:`decode_token` and from nowhere else. Every
protected endpoint depends on :func:`get_current_user`, which verifies the
signature and expiry server-side and returns the user row loaded from SQLite by
the token's subject. A client-supplied id is not merely ignored; there is no
code path that reads one.

Design notes
------------
**Login and signup return the same error for "no such email" and "wrong
password."** Distinguishing them turns the login form into an account-existence
oracle.

**A Google-only account has ``password_hash = NULL``.** Password verification
against it fails on the null check rather than by comparing against an empty
hash, which some bcrypt wrappers accept.

**Google identities are matched on ``sub``, not email.** The ``sub`` claim is
Google's stable account identifier; an email address can be changed, and
matching on it alone would let a re-registered address inherit an account. An
account is linked by email only when Google reports that address verified.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import backend.config as config
from backend import db

logger = logging.getLogger("smartdoc.auth")

# auto_error=False so a missing header raises our own AuthError -- and therefore
# our structured {"error": {...}} body -- instead of FastAPI's bare 403 detail.
_bearer = HTTPBearer(auto_error=False)

# bcrypt truncates silently at 72 bytes: with a longer password only the first
# 72 bytes are checked, so two different long passwords can both authenticate.
# Rejecting is the honest response; hashing to sidestep the limit would change
# the stored format.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_CHARS = 8


class AuthError(Exception):
    """Raised for any failure to establish an identity (mapped to HTTP 401)."""


class RegistrationError(Exception):
    """Raised for an invalid or duplicate signup (mapped to HTTP 400/409)."""


class OAuthNotConfigured(Exception):
    """Raised when a Google route is hit without Google credentials set."""


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def validate_password(password: str) -> None:
    """Reject passwords that are too short or that bcrypt would truncate."""
    if not password or len(password) < MIN_PASSWORD_CHARS:
        raise RegistrationError(
            f"Password must be at least {MIN_PASSWORD_CHARS} characters."
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise RegistrationError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes; bcrypt ignores "
            "anything beyond that, which would make a longer password weaker "
            "than it looks."
        )


def hash_password(password: str) -> str:
    """bcrypt hash, cost from ``config.BCRYPT_ROUNDS``. Salt is per-hash."""
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=config.BCRYPT_ROUNDS)
    ).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    """Constant-time check via bcrypt; False for a Google-only account."""
    if not password_hash or not password:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # A malformed stored hash must fail closed, not raise a 500.
        logger.warning("Malformed password hash encountered during verification.")
        return False


# ---------------------------------------------------------------------------
# JWTs
# ---------------------------------------------------------------------------


def assert_signing_key_usable() -> None:
    """Refuse to run with the placeholder secret unless explicitly allowed.

    A published default signing key means anyone can mint a token for any
    ``user_id``, which defeats every isolation guarantee in this phase.
    """
    is_default = config.JWT_SECRET == "dev-only-insecure-secret-change-me"
    if is_default and not config.ALLOW_INSECURE_JWT_SECRET:
        raise RuntimeError(
            "JWT_SECRET is still the development placeholder. Set a real secret "
            "in .env, or set ALLOW_INSECURE_JWT_SECRET=true to accept the risk "
            "in local development."
        )
    if is_default:
        logger.warning(
            "JWT_SECRET is the development placeholder -- tokens are forgeable. "
            "Set JWT_SECRET before exposing this API."
        )


def create_access_token(user_id: str, email: str, expires_minutes: int | None = None) -> str:
    """Sign a JWT whose subject is ``user_id``.

    The email rides along for display only. Authorization reads ``sub``; the
    user row is then loaded from SQLite, so a stale email in an old token cannot
    grant access to anything.
    """
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=expires_minutes or config.JWT_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int(expiry.timestamp()),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Verify signature and expiry, returning the claims.

    ``algorithms`` is pinned to the configured algorithm so a token declaring
    ``"alg": "none"`` -- or HS256 against an RSA public key -- cannot be accepted.
    """
    try:
        return jwt.decode(
            token,
            config.JWT_SECRET,
            algorithms=[config.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Session has expired. Please sign in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid authentication token.") from exc


# ---------------------------------------------------------------------------
# Registration and login
# ---------------------------------------------------------------------------


def normalise_email(email: str) -> str:
    cleaned = (email or "").strip().lower()
    if "@" not in cleaned or cleaned.startswith("@") or cleaned.endswith("@"):
        raise RegistrationError(f"{email!r} is not a valid email address.")
    return cleaned


def signup(email: str, password: str) -> dict:
    """Create a password account and return the user row."""
    address = normalise_email(email)
    validate_password(password)
    try:
        return db.create_user(email=address, password_hash=hash_password(password))
    except db.EmailAlreadyRegistered as exc:
        raise RegistrationError(str(exc)) from exc


def login(email: str, password: str) -> dict:
    """Verify credentials and return the user row, or raise :class:`AuthError`."""
    user = db.get_user_by_email(email or "")
    # Same message either way: a distinct "no such account" reply would let an
    # attacker enumerate registered addresses.
    if user is None or not verify_password(password, user.get("password_hash")):
        raise AuthError("Incorrect email or password.")
    return user


def upsert_google_user(claims: dict) -> dict:
    """Find or create the account behind a verified Google ``userinfo`` payload."""
    sub = (claims or {}).get("sub")
    email = (claims or {}).get("email")
    if not sub or not email:
        raise AuthError("Google did not return an account identifier and email.")

    user = db.get_user_by_google_sub(sub)
    if user:
        return user

    existing = db.get_user_by_email(email)
    if existing:
        # Only link when Google vouches for the address; otherwise an
        # unverified claim to an address would take over a password account.
        if not claims.get("email_verified", False):
            raise AuthError(
                "An account with this email already exists. Sign in with your "
                "password, or verify the address with Google first."
            )
        db.link_google_sub(existing["id"], sub)
        return db.get_user_by_id(existing["id"]) or existing

    return db.create_user(email=normalise_email(email), password_hash=None, google_sub=sub)


# ---------------------------------------------------------------------------
# Google OAuth client (authlib)
# ---------------------------------------------------------------------------

_oauth = None


def google_configured() -> bool:
    return bool(config.GOOGLE_CLIENT_ID and config.GOOGLE_CLIENT_SECRET)


def google_client():
    """Lazily build the authlib Google client from OpenID discovery."""
    global _oauth
    if not google_configured():
        raise OAuthNotConfigured(
            "Google sign-in is not configured. Set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET in .env, or use email/password sign-in."
        )
    if _oauth is None:
        from authlib.integrations.starlette_client import OAuth

        oauth = OAuth()
        oauth.register(
            name="google",
            client_id=config.GOOGLE_CLIENT_ID,
            client_secret=config.GOOGLE_CLIENT_SECRET,
            server_metadata_url=(
                "https://accounts.google.com/.well-known/openid-configuration"
            ),
            client_kwargs={"scope": "openid email profile"},
        )
        _oauth = oauth
    return _oauth.google


# ---------------------------------------------------------------------------
# The dependency that protects endpoints
# ---------------------------------------------------------------------------


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """Resolve the caller from the ``Authorization: Bearer`` token.

    Returns the SQLite user row. The row is re-read on every request rather than
    trusted from the token body, so a deleted account stops working immediately
    instead of at token expiry.
    """
    if credentials is None or not (credentials.credentials or "").strip():
        raise AuthError("Authentication required. Send an Authorization: Bearer token.")

    claims = decode_token(credentials.credentials.strip())
    user_id = claims.get("sub")
    if not user_id:
        raise AuthError("Invalid authentication token.")

    user = db.get_user_by_id(user_id)
    if user is None:
        raise AuthError("This account no longer exists.")

    # Stashed for logging/diagnostics only; nothing authorizes off request.state.
    request.state.user_id = user["id"]
    return user


def get_current_user_id(user: dict = Depends(get_current_user)) -> str:
    """The authenticated user_id -- the value every scoped query filters on."""
    return user["id"]
