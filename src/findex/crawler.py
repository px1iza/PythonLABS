
import asyncio
import logging
import random
import time

from collections.abc import AsyncIterator
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)
from urllib.robotparser import RobotFileParser

import httpx


logger = logging.getLogger("findex.crawler")

USER_AGENT = "FindexCrawler/1.0 (contact: example@example.com)"


def canonicalize_url(url: str) -> str:
    """Normalize URL to avoid duplicate pages."""
    parts = urlsplit(url)

    scheme = parts.scheme.lower()
    host = parts.netloc.lower()

    path = parts.path or "/"

    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query = urlencode(sorted(parse_qsl(parts.query)))

    return urlunsplit((scheme, host, path, query, ""))


class PageParser(HTMLParser):
    """Extract page text and links."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links = []
        self.text_parts = []
        self.skip_text = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag in ("script", "style", "noscript"):
            self.skip_text = True

        if tag == "a" and attrs.get("href"):
            link = urljoin(self.base_url, attrs["href"])
            self.links.append(canonicalize_url(link))

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip_text = False

    def handle_data(self, data):
        if not self.skip_text:
            text = data.strip()
            if text:
                self.text_parts.append(text)


@dataclass
class CrawlPage:
    url: str
    status: int
    text: str
    links: list[str]
    bytes_received: int
    elapsed: float


@dataclass
class FetchResult:
    status: int
    content: bytes
    elapsed: float


class RobotsCache:
    """Cache robots.txt rules separately for every host."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.cache = {}
        self.locks = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"

        if host not in self.locks:
            self.locks[host] = asyncio.Lock()

        async with self.locks[host]:
            if host not in self.cache:
                robots_url = f"{host}/robots.txt"

                try:
                    response = await self.client.get(robots_url)

                    parser = RobotFileParser()
                    parser.set_url(robots_url)

                    if response.status_code == 200:
                        parser.parse(response.text.splitlines())
                    else:
                        parser.parse([])

                    self.cache[host] = parser

                except httpx.HTTPError:
                    logger.exception("Could not read robots.txt: %s", host)

                    parser = RobotFileParser()
                    parser.parse([])
                    self.cache[host] = parser

        return self.cache[host].can_fetch(USER_AGENT, url)


def retry_delay(attempt: int, retry_after: str | None = None) -> float:
    """Calculate exponential backoff with jitter."""
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass

    delay = min(2 ** attempt, 30)
    return delay + random.uniform(0, 1)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> FetchResult:
    """Download a page with timeout and retry handling."""
    for attempt in range(max_retries + 1):
        started = time.perf_counter()

        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url)

            elapsed = time.perf_counter() - started

            logger.info(
                "URL=%s status=%s bytes=%s time=%.3f",
                url,
                response.status_code,
                len(response.content),
                elapsed,
            )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < max_retries:
                    delay = retry_delay(
                        attempt,
                        response.headers.get("Retry-After"),
                    )

                    logger.warning(
                        "Retry URL=%s status=%s attempt=%s delay=%.2f",
                        url,
                        response.status_code,
                        attempt + 1,
                        delay,
                    )

                    await asyncio.sleep(delay)
                    continue

            response.raise_for_status()

            return FetchResult(
                status=response.status_code,
                content=response.content,
                elapsed=elapsed,
            )

        except (TimeoutError, httpx.TimeoutException):
            if attempt >= max_retries:
                logger.error("Timeout after retries: %s", url)
                raise

            delay = retry_delay(attempt)

            logger.warning(
                "Timeout URL=%s attempt=%s delay=%.2f",
                url,
                attempt + 1,
                delay,
            )

            await asyncio.sleep(delay)

    raise RuntimeError(f"Could not fetch {url}")


async def crawl(
    start_url: str,
    max_pages: int = 200,
    concurrency: int = 5,
    per_host: int = 2,
    delay: float = 0.5,
) -> AsyncIterator[CrawlPage]:
    """Crawl allowed pages and yield them as they arrive."""
    start_url = canonicalize_url(start_url)

    queue = asyncio.Queue()
    await queue.put(start_url)

    seen = {start_url}
    scheduled = 1
    completed = 0

    global_semaphore = asyncio.Semaphore(concurrency)
    host_semaphores = {}
    host_last_request = {}

    results = asyncio.Queue()
    finished = asyncio.Event()

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=None,
    ) as client:

        robots = RobotsCache(client)

        async def worker():
            nonlocal scheduled, completed

            while True:
                try:
                    url = await asyncio.wait_for(queue.get(), timeout=1)
                except TimeoutError:
                    if completed >= scheduled:
                        return
                    continue

                try:
                    if not await robots.allowed(url):
                        logger.info("ROBOTS_DENIED URL=%s", url)
                        continue

                    host = urlsplit(url).netloc

                    if host not in host_semaphores:
                        host_semaphores[host] = asyncio.Semaphore(per_host)

                    async with global_semaphore:
                        async with host_semaphores[host]:
                            previous = host_last_request.get(host, 0)
                            wait_time = delay - (time.monotonic() - previous)

                            if wait_time > 0:
                                await asyncio.sleep(wait_time)

                            host_last_request[host] = time.monotonic()

                            result = await fetch(client, url)

                    parser = PageParser(url)
                    parser.feed(result.content.decode("utf-8", errors="replace"))

                    page = CrawlPage(
                        url=url,
                        status=result.status,
                        text=" ".join(parser.text_parts),
                        links=parser.links,
                        bytes_received=len(result.content),
                        elapsed=result.elapsed,
                    )

                    await results.put(page)

                    for link in parser.links:
                        if scheduled >= max_pages:
                            break

                        parts = urlsplit(link)

                        if parts.scheme not in ("http", "https"):
                            continue

                        if link not in seen:
                            seen.add(link)
                            await queue.put(link)
                            scheduled += 1

                except Exception:
                    logger.exception("CRAWL_ERROR URL=%s", url)

                finally:
                    completed += 1
                    queue.task_done()

        async with asyncio.TaskGroup() as group:
            for _ in range(concurrency):
                group.create_task(worker())

            while completed < scheduled:
                try:
                    page = await asyncio.wait_for(results.get(), timeout=0.2)
                    yield page
                except TimeoutError:
                    continue

            # Collect any remaining results before leaving.
            while not results.empty():
                yield results.get_nowait()

            finished.set()
