"""Canonical CLI entry point for enriched tick loading."""


def main() -> None:
    from orderflow.storage.load_ticks_enriched_cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
