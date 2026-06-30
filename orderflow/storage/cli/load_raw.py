"""Canonical CLI entry point for raw tick loading."""


def main() -> None:
    from orderflow.storage.load_ticks_raw_cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
