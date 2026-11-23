
import httpx
import pytest

from findex.crawler import (
    PageParser,
    RobotsCache,
    canonicalize_url,
    fetch,
)


def test_canonicalize_url():
    url = "HTTPS://Example.COM/docs/?z=2&a=1#section"

    result = canonicalize_url(url)

    assert result == "https://example.com/docs?a=1&z=2"


def test_canonicalize_root_url():
    assert canonicalize_url("HTTPS://Example.COM/#top") == (
        "https://example.com/"
    )


def test_page_parser_extracts_text_and_links():
    html = """
    <html>
        <head>
            <title>Test page</title>
            <script>hidden text</script>
        </head>
        <body>
            <h1>Hello world</h1>
            <p>This is a test.</p>
            <a href="/about">About</a>
        </body>
    </html>
    """

    parser = PageParser("https://example.com/")
    parser.feed(html)

    text = " ".join(parser.text_parts)

    assert "Hello world" in text
    assert "This is a test." in text
    assert "hidden text" not in text
    assert "https://example.com/about" in parser.links


@pytest.mark.asyncio
async def test_fetch_success():
    def handler(request):
        return httpx.Response(
            200,
            content=b"Hello",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        result = await fetch(client, "https://example.com")

    assert result.status == 200
    assert result.content == b"Hello"
    assert result.elapsed >= 0


@pytest.mark.asyncio
async def test_fetch_retries_after_429(monkeypatch):
    attempts = 0

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr("findex.crawler.asyncio.sleep", fake_sleep)

    def handler(request):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                request=request,
            )

        return httpx.Response(
            200,
            content=b"Success",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        result = await fetch(client, "https://example.com")

    assert attempts == 2
    assert result.status == 200
    assert result.content == b"Success"


@pytest.mark.asyncio
async def test_fetch_does_not_retry_404():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            404,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch(client, "https://example.com")

    assert attempts == 1


@pytest.mark.asyncio
async def test_robots_cache():
    robots_requests = 0

    def handler(request):
        nonlocal robots_requests
        robots_requests += 1

        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text=(
                    "User-agent: *\n"
                    "Disallow: /private\n"
                    "Allow: /\n"
                ),
                request=request,
            )

        return httpx.Response(200, request=request)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as client:
        robots = RobotsCache(client)

        allowed = await robots.allowed(
            "https://example.com/public"
        )

        denied = await robots.allowed(
            "https://example.com/private"
        )

    assert allowed is True
    assert denied is False
    assert robots_requests == 1
