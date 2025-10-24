"""
Sanity Checks for nfl_data_py → nflreadpy Migration
===================================================

Tests that the migration is working correctly:
1. Weekly player stats load for 2023-2025 without 404s
2. Injuries guard (2025 excluded or empty)
3. Downstream dtype expectations (pandas DataFrames)
"""

import sys
import pandas as pd

# Import the migrated functions
from nfl_data_utils import import_weekly_data, import_injuries_safe, import_schedules

def test_weekly_data_2023_2025():
    """Test 1: Weekly player stats load for 2023-2025 (should succeed without 404s)"""
    print("\n" + "="*70)
    print("TEST 1: Weekly Player Stats (2023-2025)")
    print("="*70)

    try:
        df = import_weekly_data([2023, 2024, 2025])

        # Check not empty
        assert not df.empty, "import_weekly_data returned empty DataFrame"
        print(f"✅ Loaded {len(df):,} player-weeks")

        # Check required columns
        required_cols = ["player_id", "player_name", "recent_team", "season", "week"]
        for col in required_cols:
            assert col in df.columns, f"Missing expected column: {col}"
        print(f"✅ All required columns present: {required_cols}")

        # Check seasons present
        seasons = sorted(df['season'].unique())
        print(f"✅ Seasons loaded: {seasons}")

        # Verify 2025 data is present
        df_2025 = df[df['season'] == 2025]
        if len(df_2025) > 0:
            print(f"✅ 2025 data loaded: {len(df_2025):,} player-weeks")
        else:
            print(f"⚠️  No 2025 data found (may not be available yet)")

        print("\n✅ TEST 1 PASSED: Weekly data loads successfully")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_injuries_2025_guard():
    """Test 2: Injuries guard (2025 excluded or empty)"""
    print("\n" + "="*70)
    print("TEST 2: Injuries 2025 Guard")
    print("="*70)

    try:
        inj = import_injuries_safe([2024, 2025])

        if inj.empty:
            print("✅ Injuries returned empty DataFrame (expected if unavailable)")
        else:
            print(f"✅ Loaded {len(inj):,} injury records")

            # Check season column exists
            assert "season" in inj.columns or inj.empty, "Missing 'season' column in injuries"
            print("✅ 'season' column present")

            # Check no 2025 data
            if not inj.empty and "season" in inj.columns:
                max_season = inj['season'].max()
                assert max_season <= 2024, f"2025 data leaked through guard: max season = {max_season}"
                print(f"✅ No 2025 injury data (max season: {max_season})")
            else:
                print("✅ Empty DataFrame, 2025 guard implicit")

        print("\n✅ TEST 2 PASSED: 2025 injury guard working")
        return True

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pandas_dtype():
    """Test 3: Downstream dtype expectations (if code assumes pandas)"""
    print("\n" + "="*70)
    print("TEST 3: DataFrame Type Verification")
    print("="*70)

    try:
        df = import_weekly_data([2024])

        # Check is pandas DataFrame
        assert isinstance(df, pd.DataFrame), f"Expected pandas DataFrame, got {type(df)}"
        print(f"✅ Returned pandas DataFrame (type: {type(df)})")

        # Check basic pandas operations work
        assert hasattr(df, 'head'), "DataFrame missing 'head' method"
        assert hasattr(df, 'groupby'), "DataFrame missing 'groupby' method"
        assert hasattr(df, 'merge'), "DataFrame missing 'merge' method"
        print("✅ Standard pandas methods available")

        # Check dtype attributes
        assert hasattr(df, 'dtypes'), "DataFrame missing 'dtypes' attribute"
        print(f"✅ DataFrame has dtypes: {len(df.dtypes)} columns")

        print("\n✅ TEST 3 PASSED: pandas DataFrame type correct")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schedules():
    """Additional Test: Schedules loading"""
    print("\n" + "="*70)
    print("ADDITIONAL TEST: Schedules Loading")
    print("="*70)

    try:
        df = import_schedules([2024, 2025])

        assert not df.empty, "import_schedules returned empty DataFrame"
        print(f"✅ Loaded {len(df):,} games")

        # Check required columns
        expected_cols = ['season', 'week', 'game_id', 'home_team', 'away_team']
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"⚠️  Some expected columns missing: {missing}")
        else:
            print(f"✅ All expected columns present")

        # Check 2025 data
        if 'season' in df.columns:
            seasons = sorted(df['season'].unique())
            print(f"✅ Seasons in schedules: {seasons}")

        print("\n✅ SCHEDULES TEST PASSED")
        return True

    except Exception as e:
        print(f"\n❌ SCHEDULES TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all sanity checks"""
    print("\n" + "="*70)
    print("NFLREADPY MIGRATION SANITY CHECKS")
    print("="*70)

    results = []

    # Run tests
    results.append(("Weekly Data 2023-2025", test_weekly_data_2023_2025()))
    results.append(("Injuries 2025 Guard", test_injuries_2025_guard()))
    results.append(("Pandas DataFrame Type", test_pandas_dtype()))
    results.append(("Schedules Loading", test_schedules()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Migration successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
