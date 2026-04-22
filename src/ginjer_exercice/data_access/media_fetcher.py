"""Téléchargeur de médias robuste pour les appels LLM multimodaux.

Télécharge les images et vidéos depuis des URLs publiques (GCS ou autre)
et retourne des ``MediaContent`` validés. Le téléchargement explicite en bytes
permet : un seul download par asset, validation de taille et MIME avant
l'appel LLM, contrôle des retries, et meilleure observabilité.

Usage::

    import httpx
    fetcher = MediaFetcher(client=httpx.Client())
    media = fetcher.download("https://storage.googleapis.com/.../image.jpg")
"""

import logging
import time
from typing import Final

import httpx

from ..exceptions import (
    MediaFetchError,
    MediaNotFoundError,
    MediaTooLargeError,
    MediaUnsupportedError,
)
from ..schemas.media import MediaContent, MediaKind

logger = logging.getLogger(__name__)

# Types MIME supportés par le pipeline
_SUPPORTED_IMAGE_MIMES: Final[frozenset[str]] = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
})

_SUPPORTED_VIDEO_MIMES: Final[frozenset[str]] = frozenset({
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/mpeg",
})

_ALL_SUPPORTED_MIMES: Final[frozenset[str]] = _SUPPORTED_IMAGE_MIMES | _SUPPORTED_VIDEO_MIMES

_DEFAULT_MAX_SIZE_BYTES: Final[int] = 50 * 1024 * 1024  # 50 MB
_DEFAULT_IMAGE_TIMEOUT: Final[float] = 30.0
_DEFAULT_VIDEO_TIMEOUT: Final[float] = 120.0
_DEFAULT_MAX_RETRIES: Final[int] = 3
_BACKOFF_BASE: Final[float] = 1.0

_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({500, 502, 503, 504})


def _infer_kind(mime_type: str) -> MediaKind:
    """Détermine ``MediaKind`` à partir du type MIME."""
    if mime_type in _SUPPORTED_IMAGE_MIMES:
        return MediaKind.IMAGE
    if mime_type in _SUPPORTED_VIDEO_MIMES:
        return MediaKind.VIDEO
    raise MediaUnsupportedError(f"Type MIME non supporté : {mime_type}")


def _extract_mime(response: httpx.Response, url: str) -> str:
    """Extrait et valide le type MIME depuis la réponse HTTP.

    Utilise le header Content-Type comme source de vérité.

    Args:
        response: Réponse HTTP.
        url: URL d'origine (pour les messages d'erreur).

    Returns:
        Type MIME nettoyé (sans paramètres charset, etc.).

    Raises:
        MediaUnsupportedError: Si le type MIME n'est pas supporté.
    """
    content_type = response.headers.get("content-type", "")
    # Extraire le type MIME sans les paramètres (e.g. "image/jpeg; charset=utf-8")
    mime_type = content_type.split(";")[0].strip().lower()

    if mime_type not in _ALL_SUPPORTED_MIMES:
        raise MediaUnsupportedError(
            f"Type MIME non supporté '{mime_type}' pour l'URL : {url}"
        )

    return mime_type


class MediaFetcher:
    """Téléchargeur de médias avec retries, validation MIME et contrôle de taille.

    Args:
        client: Instance ``httpx.Client`` injectée.
        max_size_bytes: Taille maximale autorisée (défaut : 50 MB).
        image_timeout: Timeout pour les images (défaut : 30s).
        video_timeout: Timeout pour les vidéos (défaut : 120s).
        max_retries: Nombre maximal de tentatives (défaut : 3).
    """

    def __init__(
        self,
        client: httpx.Client,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
        image_timeout: float = _DEFAULT_IMAGE_TIMEOUT,
        video_timeout: float = _DEFAULT_VIDEO_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._client = client
        self._max_size_bytes = max_size_bytes
        self._image_timeout = image_timeout
        self._video_timeout = video_timeout
        self._max_retries = max_retries

    # ── Public API ──────────────────────────────────────────────

    def download(self, url: str) -> MediaContent:
        """Télécharge un média et retourne un ``MediaContent`` validé.

        Args:
            url: URL publique du média.

        Returns:
            ``MediaContent`` validé avec les octets bruts.

        Raises:
            MediaNotFoundError: Si HTTP 404.
            MediaTooLargeError: Si le contenu dépasse la limite.
            MediaUnsupportedError: Si le type MIME n'est pas supporté.
            MediaFetchError: Pour tout autre échec après les retries.
        """
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            start = time.monotonic()
            try:
                return self._do_download(url, attempt)
            except (MediaNotFoundError, MediaTooLargeError, MediaUnsupportedError):
                # Non-retryable errors — raise immediately
                raise
            except MediaFetchError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                last_error = exc
                if attempt < self._max_retries:
                    backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "Media download failed, retrying",
                        extra={
                            "url": url,
                            "attempt": attempt,
                            "max_retries": self._max_retries,
                            "backoff_s": backoff,
                            "latency_ms": round(latency_ms, 1),
                            "error": str(exc),
                        },
                    )
                    import time as time_mod
                    time_mod.sleep(backoff)
                else:
                    logger.error(
                        "Media download failed after all retries",
                        extra={
                            "url": url,
                            "attempts": attempt,
                            "latency_ms": round(latency_ms, 1),
                            "error": str(exc),
                        },
                    )

        raise MediaFetchError(
            f"Échec du téléchargement après {self._max_retries} tentatives : {url}"
        ) from last_error

    def download_all(self, urls: list[str]) -> list[MediaContent]:
        """Télécharge une liste de médias. Les échecs sont ignorés avec un warning.

        Args:
            urls: Liste d'URLs publiques.

        Returns:
            Liste de ``MediaContent`` téléchargés avec succès (résultats partiels).
        """
        results: list[MediaContent] = []
        for url in urls:
            try:
                results.append(self.download(url))
            except (MediaFetchError, MediaUnsupportedError) as exc:
                logger.warning(
                    "Skipping failed media download",
                    extra={"url": url, "error": str(exc)},
                )
        return results


    def _do_download(self, url: str, attempt: int) -> MediaContent:
        """Effectue un téléchargement en streaming avec validation."""
        try:
            with self._client.stream("GET", url, follow_redirects=True) as response:
                if response.status_code == 404:
                    raise MediaNotFoundError(f"Média introuvable (404) : {url}")

                if response.status_code in _RETRYABLE_STATUS_CODES:
                    raise MediaFetchError(
                        f"Erreur serveur HTTP {response.status_code} pour : {url}"
                    )

                if response.status_code >= 400:
                    raise MediaFetchError(
                        f"Erreur HTTP {response.status_code} pour : {url}"
                    )

                mime_type = _extract_mime(response, url)
                kind = _infer_kind(mime_type)


                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self._max_size_bytes:
                    raise MediaTooLargeError(
                        f"Média trop volumineux ({int(content_length)} octets, "
                        f"max {self._max_size_bytes}) : {url}"
                    )

                chunks: list[bytes] = []
                total_size = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    total_size += len(chunk)
                    if total_size > self._max_size_bytes:
                        raise MediaTooLargeError(
                            f"Média trop volumineux (>{self._max_size_bytes} octets "
                            f"pendant le streaming) : {url}"
                        )
                    chunks.append(chunk)

                content = b"".join(chunks)

                start_time = time.monotonic()
                logger.info(
                    "Media downloaded",
                    extra={
                        "url": url,
                        "mime_type": mime_type,
                        "kind": kind.value,
                        "size_bytes": total_size,
                        "attempt": attempt,
                    },
                )

                return MediaContent(
                    url=url,
                    kind=kind,
                    mime_type=mime_type,
                    content=content,
                    size_bytes=total_size,
                )

        except (MediaNotFoundError, MediaTooLargeError, MediaUnsupportedError):
            raise
        except httpx.TimeoutException as exc:
            raise MediaFetchError(f"Timeout lors du téléchargement de : {url}") from exc
        except httpx.ConnectError as exc:
            raise MediaFetchError(f"Erreur de connexion pour : {url}") from exc
        except httpx.HTTPError as exc:
            raise MediaFetchError(f"Erreur HTTP pour : {url} — {exc}") from exc
