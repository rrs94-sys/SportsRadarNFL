"""
Test script to check what data is actually available from TANK01 API
"""

import requests
from phase1_config import Config

def test_tank01_api():
    """Test TANK01 API to see what data is available"""

    config = Config()
    headers = {
        'X-RapidAPI-Key': config.TANK_API_KEY,
        'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
    }

    # Test different seasons and weeks
    test_cases = [
        {"season": 2024, "week": 8, "description": "2024 Week 8"},
        {"season": 2024, "week": 7, "description": "2024 Week 7"},
        {"season": 2024, "week": 1, "description": "2024 Week 1"},
        {"season": 2025, "week": 1, "description": "2025 Week 1"},
    ]

    print("=" * 70)
    print("TESTING TANK01 API DATA AVAILABILITY")
    print("=" * 70)

    for test in test_cases:
        print(f"\n🔍 Testing: {test['description']}")
        print("-" * 70)

        url = f"{config.TANK_BASE_URL}/getNFLGamesForWeek"
        params = {
            'week': str(test['week']),
            'seasonType': 'reg',
            'season': str(test['season'])
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                body = data.get('body', {})

                if isinstance(body, dict):
                    games = body.get('games', [])
                    if games:
                        print(f"   ✅ SUCCESS - Found {len(games)} games")
                        print(f"   Sample game: {games[0].get('gameID', 'N/A')}")
                    else:
                        print(f"   ⚠️  API responded but no games found")
                        print(f"   Response keys: {list(body.keys())}")
                else:
                    print(f"   ⚠️  Unexpected response format")
                    print(f"   Response: {str(data)[:200]}")
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.reason}")

        except requests.exceptions.Timeout:
            print(f"   ❌ TIMEOUT - API did not respond in 10 seconds")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR: {str(e)}")
        except Exception as e:
            print(f"   ❌ UNEXPECTED ERROR: {str(e)}")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_tank01_api()
