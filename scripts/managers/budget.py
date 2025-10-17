import time

import numpy as np
import pandas as pd

from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.lib.errors.budget import (
    MemoryBudgetExceededError,
    RoundsBudgetExceededError,
    SamplesBudgetExceededError,
    TimeBudgetExceededError,
)
from TheCausalityGame.core.managers.budget import BudgetManager


def simulate_dataframe_memory(rows: int = 100, cols: int = 5) -> pd.DataFrame:
    """Generate a DataFrame with `rows x cols` of float64s to simulate memory use."""
    df = pd.DataFrame(
        np.random.rand(rows, cols), columns=[f"col{i}" for i in range(cols)]
    )
    return df


def main():
    # Define a test budget with small limits
    spec = BudgetSpec(
        rounds=2,
        time_s=1.0,
        samples=5,
        memory_mb=0.001,  # ~1 KB
    )

    manager = BudgetManager(spec)
    manager.start_time()

    print("✅ Initial snapshot:")
    print(manager.snapshot())

    print("\n▶ Tick round 1...")
    manager.tick_round()
    print("Snapshot:", manager.snapshot())

    print("\n▶ Charge 3 samples...")
    manager.charge_samples(3)
    print("Snapshot:", manager.snapshot())

    print("\n▶ Simulate realistic memory usage (DataFrame)...")
    df = simulate_dataframe_memory(rows=10, cols=5)
    bytes_used = df.memory_usage(deep=True).sum()
    print(f"🧠 Simulated memory usage: {bytes_used} bytes")
    try:
        manager.charge_memory(bytes_used)
        print("✅ Memory charged successfully.")
    except MemoryBudgetExceededError as e:
        print(f"🛑 Memory enforcement triggered: {e}")

    print("Snapshot:", manager.snapshot())

    print("\n▶ Sleep to exceed time...")
    time.sleep(1.1)
    try:
        manager.check_time()
    except TimeBudgetExceededError as e:
        print(f"🛑 Time enforcement triggered: {e}")

    print("\n▶ Try exceeding samples...")
    try:
        manager.charge_samples(3)
    except SamplesBudgetExceededError as e:
        print(f"🛑 Samples enforcement triggered: {e}")

    print("\n▶ Try exceeding rounds...")
    try:
        manager.tick_round()
        manager.tick_round()
    except RoundsBudgetExceededError as e:
        print(f"🛑 Rounds enforcement triggered: {e}")

    print("\n✅ Final snapshot:")
    print(manager.snapshot())


if __name__ == "__main__":
    main()
