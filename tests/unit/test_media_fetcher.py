"""Tests unitaires pour MediaFetcher.

Tous les appels HTTP sont mockés avec ``respx`` pour couvrir
les cas de succès, erreurs, retries et validation.
"""

import pytest
import httpx
import respx

from src.ginjer_exercice.data_access.media_fetcher import MediaFetcher
from src.ginjer_exercice.schemas.media import MediaContent, MediaKind
from src.ginjer_exercice.exceptions import (
    MediaFetchError,
    MediaNotFoundError,
    MediaTooLargeError,
    MediaUnsupportedError,
)


@pytest.fixture
def fetcher():
    """MediaFetcher avec un client httpx non-mocké (respx interceptera)."""
    client = httpx.Client()
    return MediaFetcher(
        client=client,
        max_size_bytes=1024,  # 1 KB pour les tests
        image_timeout=5.0,
        video_timeout=10.0,
        max_retries=3,
    )


# ── download() — success ──────────────────────────────────────


class TestDownloadSuccess:
    """Tests de téléchargement réussi."""

    @respx.mock
    def test_image_200_returns_media_content(self, fetcher):
        """Un 200 avec image/jpeg retourne un MediaContent IMAGE."""
        content = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        url = "https://storage.googleapis.com/test/image.jpg"

        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=content,
                headers={"content-type": "image/jpeg", "content-length": str(len(content))},
            )
        )

        result = fetcher.download(url)
        assert isinstance(result, MediaContent)
        assert result.kind == MediaKind.IMAGE
        assert result.mime_type == "image/jpeg"
        assert result.content == content
        assert result.size_bytes == len(content)
        assert result.url == url

    @respx.mock
    def test_video_200_returns_media_content(self, fetcher):
        """Un 200 avec video/mp4 retourne un MediaContent VIDEO."""
        content = b"\x00\x00\x00\x1c" + b"\x00" * 50
        url = "https://storage.googleapis.com/test/video.mp4"

        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=content,
                headers={"content-type": "video/mp4"},
            )
        )

        result = fetcher.download(url)
        assert result.kind == MediaKind.VIDEO
        assert result.mime_type == "video/mp4"

    @respx.mock
    def test_content_type_with_charset(self, fetcher):
        """Le type MIME est extrait correctement même avec un paramètre charset."""
        content = b"\x89PNG\r\n" + b"\x00" * 50
        url = "https://example.com/image.png"

        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=content,
                headers={"content-type": "image/png; charset=utf-8"},
            )
        )

        result = fetcher.download(url)
        assert result.mime_type == "image/png"
        assert result.kind == MediaKind.IMAGE


# ── download() — errors ───────────────────────────────────────


class TestDownloadErrors:
    """Tests des erreurs de téléchargement."""

    @respx.mock
    def test_404_raises_media_not_found(self, fetcher):
        """HTTP 404 lève MediaNotFoundError (sans retry)."""
        url = "https://example.com/missing.jpg"
        respx.get(url).mock(return_value=httpx.Response(404))

        with pytest.raises(MediaNotFoundError):
            fetcher.download(url)

    @respx.mock
    def test_oversized_content_length_raises(self, fetcher):
        """Un Content-Length > max lève MediaTooLargeError (sans retry)."""
        url = "https://example.com/huge.jpg"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=b"x",
                headers={"content-type": "image/jpeg", "content-length": "999999"},
            )
        )

        with pytest.raises(MediaTooLargeError):
            fetcher.download(url)

    @respx.mock
    def test_oversized_during_streaming_raises(self, fetcher):
        """Un contenu qui dépasse la taille max pendant le streaming lève MediaTooLargeError."""
        url = "https://example.com/big.jpg"
        content = b"x" * 2048  # 2 KB > max 1 KB
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=content,
                headers={"content-type": "image/jpeg"},
            )
        )

        with pytest.raises(MediaTooLargeError):
            fetcher.download(url)

    @respx.mock
    def test_unsupported_mime_raises(self, fetcher):
        """Un type MIME non supporté lève MediaUnsupportedError (sans retry)."""
        url = "https://example.com/doc.pdf"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=b"pdf content",
                headers={"content-type": "application/pdf"},
            )
        )

        with pytest.raises(MediaUnsupportedError):
            fetcher.download(url)


# ── download() — retries ──────────────────────────────────────


class TestDownloadRetries:
    """Tests du mécanisme de retry."""

    @respx.mock
    def test_retry_on_500_then_success(self, fetcher):
        """Un 500 suivi d'un 200 retourne le résultat sans erreur."""
        url = "https://example.com/retry.jpg"
        content = b"\xff" * 100

        route = respx.get(url)
        route.side_effect = [
            httpx.Response(500),
            httpx.Response(
                200,
                content=content,
                headers={"content-type": "image/jpeg"},
            ),
        ]

        result = fetcher.download(url)
        assert result.kind == MediaKind.IMAGE
        assert route.call_count == 2

    @respx.mock
    def test_fail_after_max_retries(self, fetcher):
        """Des 500 répétés lèvent MediaFetchError après toutes les tentatives."""
        url = "https://example.com/always_fail.jpg"

        route = respx.get(url)
        route.side_effect = [
            httpx.Response(500),
            httpx.Response(502),
            httpx.Response(503),
        ]

        with pytest.raises(MediaFetchError):
            fetcher.download(url)
        assert route.call_count == 3


# ── download_all() ────────────────────────────────────────────


class TestDownloadAll:
    """Tests pour download_all (résultats partiels)."""

    @respx.mock
    def test_mixed_results_returns_partial(self, fetcher):
        """download_all([ok, missing, ok]) retourne 2 résultats et log un warning."""
        content = b"\xff" * 50

        url_ok_1 = "https://example.com/ok1.jpg"
        url_missing = "https://example.com/missing.jpg"
        url_ok_2 = "https://example.com/ok2.png"

        respx.get(url_ok_1).mock(
            return_value=httpx.Response(
                200, content=content, headers={"content-type": "image/jpeg"}
            )
        )
        respx.get(url_missing).mock(return_value=httpx.Response(404))
        respx.get(url_ok_2).mock(
            return_value=httpx.Response(
                200, content=content, headers={"content-type": "image/png"}
            )
        )

        results = fetcher.download_all([url_ok_1, url_missing, url_ok_2])
        assert len(results) == 2
        assert results[0].url == url_ok_1
        assert results[1].url == url_ok_2

    @respx.mock
    def test_all_fail_returns_empty(self, fetcher):
        """download_all avec tous les échecs retourne une liste vide."""
        url = "https://example.com/fail.jpg"
        respx.get(url).mock(return_value=httpx.Response(404))

        results = fetcher.download_all([url])
        assert results == []
