"""
Test what data is actually available from nfl_data_py
"""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime

def test_nfl_data_py():
    """Test what seasons are available from nfl_data_py"""

    print("=" * 70)
    print("TESTING NFL_DATA_PY DATA AVAILABILITY")
    print("=" * 70)
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print()

    # Test different seasons
    test_seasons = [2025, 2024, 2023, 2022]

    for season in test_seasons:
        print(f"\n🔍 Testing Season {season}")
        print("-" * 70)

        try:
            # Try to import weekly data
            data = nfl.import_weekly_data([season], downcast=False)

            if data is not None and not data.empty:
                weeks = sorted(data['week'].unique())
                players = data['player_id'].nunique()
                rows = len(data)

                print(f"   ✅ SUCCESS - Data available")
                print(f"   Weeks: {weeks}")
                print(f"   Players: {players:,}")
                print(f"   Total rows: {rows:,}")
            else:
                print(f"   ⚠️  API responded but returned empty data")

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "Not Found" in error_msg:
                print(f"   ❌ NOT AVAILABLE - Data not published yet")
            elif "HTTP" in error_msg:
                print(f"   ❌ HTTP ERROR - {error_msg[:100]}")
            else:
                print(f"   ❌ ERROR - {error_msg[:100]}")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    print("\n💡 RECOMMENDATION:")
    print("   Use the most recent season with available data")
    print("   for training and predictions.")
    print("=" * 70)

if __name__ == "__main__":
    test_nfl_data_py()
