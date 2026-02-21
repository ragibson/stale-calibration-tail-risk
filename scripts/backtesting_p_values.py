from src.backtesting import backtesting_p_value

if __name__ == "__main__":
    for trials in [25, 50]:
        for num_failures in range(trials // 5 + 1):
            print(f"Failures: {num_failures:<2} / {trials:<2} -> "
                  f"p={backtesting_p_value(num_failures, trials, confidence_level=0.95):.1%}")
        print()
