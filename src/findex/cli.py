
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from findex.crawler import crawl
from findex.lab3 import BM25, TfIdf, open_index, save_index, search
from findex.parallel import build_index_parallel


app = typer.Typer(
    name="findex",
    help="A command-line full-text search engine.",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger("findex")


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase logging verbosity.",
        ),
    ] = 0,
) -> None:
    """Configure logging for the CLI."""
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


@app.command("index")
def index_command(
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Directory with text documents."),
    ] = Path("data"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output index file."),
    ] = Path("index_v4.bin"),
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Number of workers."),
    ] = 1,
    executor: Annotated[
        Literal["serial", "threads", "processes"],
        typer.Option(
            "--executor",
            help="Execution mode: serial, threads, or processes.",
        ),
    ] = "serial",
) -> None:
    """Build an inverted index from text documents."""
    if not data.is_dir():
        typer.echo(f"Error: directory '{data}' does not exist.", err=True)
        raise typer.Exit(code=1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Building index...", total=1)

            index = build_index_parallel(
                data,
                workers=workers,
                executor=executor,
            )

            save_index(index, output)
            progress.update(task, completed=1)

        console.print(f"[green]Index saved to:[/green] {output}")
        console.print(f"Documents: {index.num_docs}")
        console.print(f"Unique terms: {len(index)}")

        logger.info("Index successfully created.")

    except (OSError, ValueError, TypeError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("search")
def search_command(
    query: Annotated[str, typer.Argument(help="Search query.")],
    index_path: Annotated[
        Path,
        typer.Option("--index", help="Path to the index file."),
    ] = Path("index_v4.bin"),
    scorer: Annotated[
        Literal["tfidf", "bm25"],
        typer.Option("--scorer", help="Ranking algorithm."),
    ] = "bm25",
    top: Annotated[
        int,
        typer.Option("--top", "-k", min=1, help="Number of results."),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON."),
    ] = False,
) -> None:
    """Search documents in the index."""
    if not index_path.is_file():
        typer.echo(f"Error: index file '{index_path}' does not exist.", err=True)
        raise typer.Exit(code=1)

    scoring = TfIdf() if scorer == "tfidf" else BM25()

    try:
        with open_index(index_path) as index:
            results = search(
                index,
                query,
                scorer=scoring,
                top_k=top,
            )

        if json_output:
            output = [
                {
                    "rank": rank,
                    "document": result.filename,
                    "score": result.score,
                    "snippet": result.snippet,
                }
                for rank, result in enumerate(results, start=1)
            ]

            json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            return

        table = Table(title="Search Results")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Document", style="green")
        table.add_column("Score", justify="right")
        table.add_column("Snippet")

        for rank, result in enumerate(results, start=1):
            table.add_row(
                str(rank),
                result.filename,
                f"{result.score:.4f}",
                result.snippet,
            )

        console.print(table)

        logger.info("Search completed. Found %d results.", len(results))

    except (OSError, ValueError, TypeError, KeyError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("stats")
def stats_command(
    index_path: Annotated[
        Path,
        typer.Argument(help="Path to the index file."),
    ] = Path("index_v4.bin"),
) -> None:
    """Display index statistics."""
    if not index_path.is_file():
        typer.echo(f"Error: index file '{index_path}' does not exist.", err=True)
        raise typer.Exit(code=1)

    try:
        with open_index(index_path) as index:
            table = Table(title="Index Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

            table.add_row("Documents", str(index.num_docs))
            table.add_row("Unique terms", str(len(index)))
            table.add_row(
                "Average document length",
                f"{index.avg_doc_length:.2f}",
            )

            console.print(table)

    except (OSError, ValueError, TypeError, KeyError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("crawl")
def crawl_command(
    start_url: Annotated[
        str,
        typer.Argument(help="Starting URL for crawling."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSONL file."),
    ] = Path("data/crawled.jsonl"),
    max_pages: Annotated[
        int,
        typer.Option("--max-pages", min=1, help="Maximum number of pages."),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", min=1, help="Number of workers."),
    ] = 5,
) -> None:
    """Crawl a website and save pages to JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)

    crawl_logger = logging.getLogger("findex.crawler")
    file_handler = logging.FileHandler("crawl.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    )
    crawl_logger.addHandler(file_handler)

    async def run_crawler() -> None:
        started = time.perf_counter()
        pages_count = 0
        errors_count = 0

        try:
            with output.open("w", encoding="utf-8") as file:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed} pages"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:

                    task = progress.add_task(
                        "Crawling website...",
                        total=max_pages,
                    )

                    async for page in crawl(
                        start_url=start_url,
                        max_pages=max_pages,
                        concurrency=concurrency,
                    ):
                        record = {
                            "url": page.url,
                            "status": page.status,
                            "text": page.text,
                            "links": page.links,
                            "bytes": page.bytes_received,
                            "elapsed": page.elapsed,
                        }

                        file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        file.flush()

                        pages_count += 1

                        elapsed = time.perf_counter() - started
                        speed = pages_count / elapsed if elapsed > 0 else 0

                        progress.update(
                            task,
                            completed=pages_count,
                            description=f"Crawling: {speed:.2f} pages/sec",
                        )

        finally:
            crawl_logger.removeHandler(file_handler)
            file_handler.close()

        elapsed = time.perf_counter() - started
        speed = pages_count / elapsed if elapsed > 0 else 0

        console.print()
        console.print("[green]Crawling completed![/green]")

        table = Table(title="Crawl Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Pages collected", str(pages_count))
        table.add_row("Errors", str(errors_count))
        table.add_row("Time", f"{elapsed:.2f} seconds")
        table.add_row("Speed", f"{speed:.2f} pages/sec")
        table.add_row("Output", str(output))
        table.add_row("Log file", "crawl.log")

        console.print(table)

    try:
        asyncio.run(run_crawler())

    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("serve")
def serve_command(
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="HTTP server port."),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Number of server workers."),
    ] = 1,
) -> None:
    """Start the Findex web application."""
    import uvicorn

    uvicorn.run(
        "findex.web.app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info",
    )
