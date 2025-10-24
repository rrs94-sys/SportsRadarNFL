"""
Test which TANK01 API endpoints are actually accessible
"""

import requests
from phase1_config import Config

def test_endpoints():
    """Test different TANK01 endpoints to see what's accessible"""

    config = Config()
    headers = {
        'X-RapidAPI-Key': config.TANK_API_KEY,
        'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
    }

    endpoints = [
        {
            "name": "getNFLNews",
            "url": f"{config.TANK_BASE_URL}/getNFLNews",
            "params": {"fantasyNews": "true", "maxItems": "10"}
        },
        {
            "name": "getNFLGamesForWeek",
            "url": f"{config.TANK_BASE_URL}/getNFLGamesForWeek",
            "params": {"week": "8", "seasonType": "reg", "season": "2024"}
        },
        {
            "name": "getNFLBoxScore",
            "url": f"{config.TANK_BASE_URL}/getNFLBoxScore",
            "params": {"gameID": "20241006_DEN@LV"}
        },
        {
            "name": "getNFLTeams",
            "url": f"{config.TANK_BASE_URL}/getNFLTeams",
            "params": {}
        },
        {
            "name": "getNFLScoreboard",
            "url": f"{config.TANK_BASE_URL}/getNFLScoreboard",
            "params": {}
        }
    ]

    print("=" * 70)
    print("TESTING TANK01 API ENDPOINT ACCESS")
    print("=" * 70)
    print(f"\nAPI Key: {config.TANK_API_KEY[:20]}...")
    print(f"Base URL: {config.TANK_BASE_URL}")
    print()

    for endpoint in endpoints:
        print(f"\n🔍 Testing: {endpoint['name']}")
        print("-" * 70)

        try:
            response = requests.get(
                endpoint['url'],
                headers=headers,
                params=endpoint['params'],
                timeout=10
            )

            print(f"   Status: {response.status_code} {response.reason}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ SUCCESS - Endpoint is accessible")
                    print(f"   Response keys: {list(data.keys())}")

                    # Show sample data structure
                    body = data.get('body', {})
                    if isinstance(body, list) and body:
                        print(f"   Body type: list with {len(body)} items")
                    elif isinstance(body, dict):
                        print(f"   Body keys: {list(body.keys())[:5]}")
                except:
                    print(f"   Response: {response.text[:200]}")
            elif response.status_code == 403:
                print(f"   ❌ FORBIDDEN - No access to this endpoint")
            elif response.status_code == 404:
                print(f"   ❌ NOT FOUND - Endpoint doesn't exist")
            elif response.status_code == 504:
                print(f"   ❌ TIMEOUT - Gateway timeout")
            else:
                print(f"   ❌ ERROR - {response.text[:200]}")

        except requests.exceptions.Timeout:
            print(f"   ❌ TIMEOUT - Request timed out after 10s")
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_endpoints()
