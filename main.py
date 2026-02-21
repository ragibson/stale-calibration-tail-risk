"""Generate all presentation figures and run a few calculations in scripts/ relevant to the presentation."""

from scripts import presentation_plots, rolling_coverage_test, typical_parameter_move_test

if __name__ == "__main__":
    print("-- Generating all presentation figures --")
    presentation_plots.main()

    print("-- Running coverage test from our choice of flexible probability half-lives --")
    rolling_coverage_test.main()

    print("-- Running typical parameter move test on AAPL --")
    typical_parameter_move_test.main()
