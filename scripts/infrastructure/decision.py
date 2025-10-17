from TheCausalityGame.core.infrastructure.decisions import Decision, Experiment


def main():
    print("=== Testing Decision Manager ===\n")

    # Create an experiment decision
    decision = Decision.experiment()
    print("Created experiment decision:", decision)

    # Add experiments
    decision.add_experiment(treatment={"X": 1}, n=10)
    decision.add_experiment(treatment={"X": 0, "Y": 1}, n=5)
    print("After adding experiments:", decision)

    # Extend with mixed formats
    decision.extend(
        [
            ({"X": 1, "Z": 0}, 20),
            Experiment(treatment=None, n=15),  # observational
        ]
    )
    print("After extending with mixed formats:")
    for i, exp in enumerate(decision.experiments, 1):
        print(f"  {i}. {exp}")

    # Try making an answer decision
    answer = Decision.answer()
    print("\nCreated answer decision:", answer)

    # Confirm boolean flags
    print("Is experiment decision? ", decision.is_experiment)
    print("Is answer decision?     ", answer.is_answer)

    # Try incorrect usage (should raise)
    try:
        answer.add_experiment({"X": 1}, 10)
    except Exception as e:
        print("\n[Expected Exception] Cannot add experiment to answer decision:")
        print(" ", type(e).__name__, "-", e)


if __name__ == "__main__":
    main()
