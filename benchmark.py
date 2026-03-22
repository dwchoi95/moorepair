import argparse

from src.benchmark import BenchmarkBuilder, BenchmarkVerifier


class BenchmarkCLI:
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Benchmark management entrypoint")
        subparsers = parser.add_subparsers(dest="command", required=True)

        build_parser = subparsers.add_parser("build", help="Build benchmark datasets")
        build_parser.add_argument("--language", type=str, default=None)
        build_parser.add_argument("--min", type=int, default=20, dest="min_count")

        verify_parser = subparsers.add_parser("verify", help="Verify benchmark verdicts")
        verify_parser.add_argument("--problem", type=str, default=None)

        return parser

    @classmethod
    def run(cls) -> None:
        parser = cls.build_parser()
        args = parser.parse_args()

        if args.command == "build":
            BenchmarkBuilder.run(
                language=args.language,
                min_count=args.min_count,
            )
            return

        if args.command == "verify":
            BenchmarkVerifier.run(
                problem=args.problem,
            )
            return


if __name__ == "__main__":
    BenchmarkCLI.run()
