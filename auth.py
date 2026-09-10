# SPDX-FileCopyrightText: 2026 lee0G21
# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


HEX_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


class AuthError(ValueError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class AuthChallenge:
    value: str
    client_ip: str
    expires_at: float


@dataclass
class AuthSession:
    signing_key: bytes
    expires_at: float
    used_nonces: dict[str, float] = field(default_factory=dict)


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float) -> bool:
        events = self.events[key]
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        return True

    def discard(self, key: str) -> None:
        self.events.pop(key, None)


class AuthManager:
    """短期匿名会话、请求签名、防重放和双维度限流。"""

    def __init__(
        self,
        challenge_ttl_seconds: int,
        token_ttl_seconds: int,
        timestamp_skew_seconds: int,
        challenges_per_minute: int,
        identify_per_ip_per_minute: int,
        identify_per_token_per_minute: int,
    ):
        self.challenge_ttl_seconds = challenge_ttl_seconds
        self.token_ttl_seconds = token_ttl_seconds
        self.timestamp_skew_seconds = timestamp_skew_seconds
        self.challenges: dict[str, AuthChallenge] = {}
        self.sessions: dict[str, AuthSession] = {}
        self.challenge_limiter = SlidingWindowLimiter(challenges_per_minute)
        self.ip_limiter = SlidingWindowLimiter(identify_per_ip_per_minute)
        self.token_limiter = SlidingWindowLimiter(
            identify_per_token_per_minute
        )
        self.lock = threading.Lock()

    @staticmethod
    def token_digest(access_token: str) -> str:
        return hashlib.sha256(access_token.encode("utf-8")).hexdigest()

    @staticmethod
    def canonical_identify_request(
        timestamp: int,
        nonce: str,
        content_sha256: str,
        top_k: int,
    ) -> bytes:
        return (
            "POST\n"
            "/api/identify\n"
            f"top_k={top_k}\n"
            f"{timestamp}\n"
            f"{nonce}\n"
            f"{content_sha256.lower()}"
        ).encode("utf-8")

    def _cleanup(self, now: float) -> None:
        for challenge_id in [
            challenge_id
            for challenge_id, challenge in self.challenges.items()
            if challenge.expires_at <= now
        ]:
            self.challenges.pop(challenge_id, None)

        for token_key in [
            token_key
            for token_key, session in self.sessions.items()
            if session.expires_at <= now
        ]:
            self.sessions.pop(token_key, None)
            self.token_limiter.discard(token_key)

    def issue_challenge(
        self,
        client_ip: str,
        now: float | None = None,
    ) -> dict[str, str | int]:
        current = time.time() if now is None else now
        with self.lock:
            self._cleanup(current)
            if not self.challenge_limiter.allow(client_ip, current):
                raise AuthError(
                    429,
                    "RATE_LIMITED",
                    "当前请求较多，请稍后重试",
                )
            challenge_id = secrets.token_urlsafe(18)
            challenge_value = secrets.token_urlsafe(32)
            self.challenges[challenge_id] = AuthChallenge(
                value=challenge_value,
                client_ip=client_ip,
                expires_at=current + self.challenge_ttl_seconds,
            )
        return {
            "challenge_id": challenge_id,
            "challenge": challenge_value,
            "expires_in": self.challenge_ttl_seconds,
        }

    def exchange_challenge(
        self,
        client_ip: str,
        challenge_id: str,
        challenge_value: str,
        now: float | None = None,
    ) -> dict[str, str | int]:
        current = time.time() if now is None else now
        with self.lock:
            self._cleanup(current)
            challenge = self.challenges.pop(challenge_id, None)
            if (
                challenge is None
                or challenge.expires_at <= current
                or challenge.client_ip != client_ip
                or not hmac.compare_digest(challenge.value, challenge_value)
            ):
                raise AuthError(
                    401,
                    "INVALID_CHALLENGE",
                    "请求验证失败，请重试",
                )

            access_token = secrets.token_urlsafe(32)
            signing_key_hex = secrets.token_hex(32)
            token_key = self.token_digest(access_token)
            self.sessions[token_key] = AuthSession(
                signing_key=bytes.fromhex(signing_key_hex),
                expires_at=current + self.token_ttl_seconds,
            )
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "signing_key": signing_key_hex,
            "expires_in": self.token_ttl_seconds,
        }

    def authorize_identify(
        self,
        client_ip: str,
        authorization: str,
        timestamp_value: str,
        nonce: str,
        content_sha256: str,
        signature: str,
        top_k: int,
        now: float | None = None,
    ) -> str:
        current = time.time() if now is None else now
        if not authorization.startswith("Bearer "):
            raise AuthError(401, "AUTH_REQUIRED", "请求验证失败，请重试")
        access_token = authorization[len("Bearer "):].strip()
        if not access_token or " " in access_token:
            raise AuthError(401, "INVALID_AUTH", "请求验证失败，请重试")
        try:
            timestamp = int(timestamp_value)
        except (TypeError, ValueError) as exc:
            raise AuthError(
                401,
                "INVALID_AUTH",
                "请求验证失败，请重试",
            ) from exc
        if not NONCE_PATTERN.fullmatch(nonce):
            raise AuthError(401, "INVALID_AUTH", "请求验证失败，请重试")
        if not HEX_SHA256_PATTERN.fullmatch(content_sha256):
            raise AuthError(401, "INVALID_AUTH", "请求验证失败，请重试")
        if not HEX_SHA256_PATTERN.fullmatch(signature):
            raise AuthError(401, "INVALID_AUTH", "请求验证失败，请重试")

        token_key = self.token_digest(access_token)
        with self.lock:
            self._cleanup(current)
            if not self.ip_limiter.allow(client_ip, current):
                raise AuthError(
                    429,
                    "RATE_LIMITED",
                    "当前请求较多，请稍后重试",
                )
            session = self.sessions.get(token_key)
            if session is None:
                raise AuthError(
                    401,
                    "TOKEN_EXPIRED",
                    "请求验证已失效，请重试",
                )
            if abs(current - timestamp) > self.timestamp_skew_seconds:
                raise AuthError(
                    401,
                    "INVALID_AUTH",
                    "请求验证失败，请重试",
                )

            for used_nonce in [
                used_nonce
                for used_nonce, expires_at in session.used_nonces.items()
                if expires_at <= current
            ]:
                session.used_nonces.pop(used_nonce, None)
            if nonce in session.used_nonces:
                raise AuthError(
                    409,
                    "REPLAY_DETECTED",
                    "请求已失效，请重试",
                )

            canonical = self.canonical_identify_request(
                timestamp,
                nonce,
                content_sha256,
                top_k,
            )
            expected_signature = hmac.new(
                session.signing_key,
                canonical,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(
                expected_signature,
                signature.lower(),
            ):
                raise AuthError(
                    401,
                    "INVALID_AUTH",
                    "请求验证失败，请重试",
                )
            if not self.token_limiter.allow(token_key, current):
                raise AuthError(
                    429,
                    "RATE_LIMITED",
                    "当前请求较多，请稍后重试",
                )
            session.used_nonces[nonce] = (
                current + self.timestamp_skew_seconds * 2
            )
        return content_sha256.lower()


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
