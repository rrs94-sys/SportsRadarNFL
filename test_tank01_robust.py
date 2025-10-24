"""
Unit Tests for Production-Grade TANK01 Client
==============================================

Tests cover:
1. Retry/backoff/jitter behavior (mock 504 → success)
2. 429 with Retry-After header handling
3. Disk cache and sample file fallback
4. Schema parity with historical data
5. Week guard (won't request future weeks)
6. Host failover logic
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
import pandas as pd
import json
import time
from pathlib import Path
import tempfile
import shutil

# Import the module to test
from tank01_client_robust import (
    RobustTank01Client,
    Tank01FieldMapping,
    Tank01NetworkError,
    Tank01ValidationError,
    get_latest_completed_week_2025,
    load_2025_week,
    _validate_schema,
    _enforce_dtypes,
)


class TestRetryBackoffJitter(unittest.TestCase):
    """Test retry logic with exponential backoff and jitter"""

    @patch('tank01_client_robust.requests.Session')
    def test_504_retry_success(self, mock_session_class):
        """Test that 504 errors are retried and eventually succeed"""
        client = RobustTank01Client()

        # Mock session
        mock_session = MagicMock()
        client.session = mock_session

        # First two attempts: 504, third attempt: success
        mock_response_504 = Mock()
        mock_response_504.status_code = 504
        mock_response_504.raise_for_status.side_effect = Exception("504 Gateway Timeout")

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'body': {'test': 'data'}}
        mock_response_success.raise_for_status.return_value = None

        mock_session.get.side_effect = [
            mock_response_504,  # Attempt 1: 504
            mock_response_504,  # Attempt 2: 504
            mock_response_success  # Attempt 3: Success
        ]

        # Make request
        result = client.fetch_tank01('/test', {'param': 'value'})

        # Verify we got the successful response
        self.assertEqual(result, {'body': {'test': 'data'}})

        # Verify we made 3 attempts
        self.assertEqual(mock_session.get.call_count, 3)

    @patch('tank01_client_robust.time.sleep')
    @patch('tank01_client_robust.requests.Session')
    def test_exponential_backoff_sequence(self, mock_session_class, mock_sleep):
        """Test that backoff sequence follows [0.5, 1, 2, 4, 8]"""
        client = RobustTank01Client()

        mock_session = MagicMock()
        client.session = mock_session

        # All attempts fail with 503
        mock_response_503 = Mock()
        mock_response_503.status_code = 503
        mock_response_503.raise_for_status.side_effect = Exception("503 Service Unavailable")

        mock_session.get.return_value = mock_response_503

        # Try to fetch (will exhaust retries)
        try:
            client.fetch_tank01('/test', {'param': 'value'})
        except:
            pass

        # Check sleep was called with increasing backoffs (with jitter)
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]

        # Should be approximately [0.5, 1, 2, 4] (4 sleeps for 5 attempts)
        self.assertEqual(len(sleep_calls), 4)

        # Each should be in range of base ± 30% jitter
        expected_bases = [0.5, 1, 2, 4]
        for actual, base in zip(sleep_calls, expected_bases):
            self.assertGreaterEqual(actual, base)
            self.assertLessEqual(actual, base * 1.3)


class TestRateLimitHandling(unittest.TestCase):
    """Test 429 rate limit with Retry-After header"""

    @patch('tank01_client_robust.time.sleep')
    @patch('tank01_client_robust.requests.Session')
    def test_429_retry_after_header(self, mock_session_class, mock_sleep):
        """Test that Retry-After header is respected for 429"""
        client = RobustTank01Client()

        mock_session = MagicMock()
        client.session = mock_session

        # First attempt: 429 with Retry-After: 5
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {'Retry-After': '5'}
        mock_response_429.raise_for_status.side_effect = Exception("429 Too Many Requests")

        # Second attempt: success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'body': {'test': 'data'}}
        mock_response_success.raise_for_status.return_value = None

        mock_session.get.side_effect = [mock_response_429, mock_response_success]

        # Make request
        result = client.fetch_tank01('/test', {'param': 'value'})

        # Verify Retry-After value was used
        mock_sleep.assert_called()
        sleep_value = mock_sleep.call_args[0][0]
        self.assertEqual(sleep_value, 5.0)


class TestDiskCacheAndFallback(unittest.TestCase):
    """Test disk cache and sample file fallback"""

    def setUp(self):
        """Create temporary directories for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.sample_dir = Path(self.temp_dir) / "samples"
        self.cache_dir.mkdir(parents=True)
        self.sample_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directories"""
        shutil.rmtree(self.temp_dir)

    @patch('tank01_client_robust.Config')
    def test_cache_hit(self, mock_config):
        """Test that cached responses are used"""
        # Setup mock config
        mock_config.TANK_API_KEY = "test_key"
        mock_config.TANK_BASE_URL = "https://test.com"
        mock_config.CACHE_DIR = str(self.cache_dir)

        client = RobustTank01Client(mock_config)
        client.cache_dir = self.cache_dir / "tank01"
        client.cache_dir.mkdir()

        # Pre-populate cache
        cache_data = {'body': {'cached': 'data'}}
        cache_file = client.cache_dir / client._get_cache_key('/test', {'param': 'value'})
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        # Fetch (should hit cache)
        result = client.fetch_tank01('/test', {'param': 'value'})

        self.assertEqual(result, cache_data)

    @patch('tank01_client_robust.Config')
    @patch('tank01_client_robust.requests.Session')
    def test_sample_fallback(self, mock_session_class, mock_config):
        """Test fallback to sample file when API fails"""
        # Setup mock config
        mock_config.TANK_API_KEY = "test_key"
        mock_config.TANK_BASE_URL = "https://test.com"
        mock_config.CACHE_DIR = str(self.cache_dir)

        client = RobustTank01Client(mock_config)
        client.cache_dir = self.cache_dir / "tank01"
        client.sample_dir = self.sample_dir
        client.cache_dir.mkdir()

        # Create sample file
        sample_data = {'body': {'sample': 'data'}}
        sample_file = client.sample_dir / client._get_cache_key('/test', {'param': 'value'})
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f)

        # Mock all API attempts to fail
        mock_session = MagicMock()
        client.session = mock_session

        mock_response_504 = Mock()
        mock_response_504.status_code = 504
        mock_response_504.raise_for_status.side_effect = Exception("504")

        mock_session.get.return_value = mock_response_504

        # Fetch (should fall back to sample)
        result = client.fetch_tank01('/test', {'param': 'value'})

        self.assertEqual(result, sample_data)


class TestSchemaValidation(unittest.TestCase):
    """Test schema parity with historical data"""

    def test_required_columns_present(self):
        """Test that all required columns are in mapping"""
        required = Tank01FieldMapping.get_required_columns()

        # Check critical columns
        self.assertIn('player_id', required)
        self.assertIn('player_name', required)
        self.assertIn('position', required)
        self.assertIn('team', required)
        self.assertIn('season', required)
        self.assertIn('week', required)
        self.assertIn('passing_yards', required)
        self.assertIn('rushing_yards', required)
        self.assertIn('receiving_yards', required)

    def test_validate_schema_success(self):
        """Test schema validation passes for valid DataFrame"""
        # Create valid DataFrame
        df = pd.DataFrame({
            'player_id': ['123'],
            'player_name': ['Test Player'],
            'position': ['QB'],
            'team': ['KC'],
            'season': [2025],
            'week': [1],
            'game_id': ['game123'],
            'attempts': [30],
            'completions': [20],
            'passing_yards': [250],
            'passing_tds': [2],
            'interceptions': [0],
            'sacks': [1],
            'carries': [0],
            'rushing_yards': [0],
            'rushing_tds': [0],
            'targets': [0],
            'receptions': [0],
            'receiving_yards': [0],
            'receiving_tds': [0],
        })

        # Should not raise
        try:
            _validate_schema(df)
        except Tank01ValidationError as e:
            self.fail(f"Validation failed unexpectedly: {e}")

    def test_validate_schema_missing_columns(self):
        """Test schema validation fails for missing columns"""
        # Create incomplete DataFrame
        df = pd.DataFrame({
            'player_id': ['123'],
            'player_name': ['Test Player'],
            # Missing many required columns
        })

        # Should raise
        with self.assertRaises(Tank01ValidationError):
            _validate_schema(df)

    def test_enforce_dtypes(self):
        """Test that dtypes are enforced correctly"""
        df = pd.DataFrame({
            'player_id': [123],  # Should become string
            'player_name': ['Test'],
            'position': ['QB'],
            'team': ['KC'],
            'season': ['2025'],  # Should become int
            'week': ['1'],  # Should become int
            'passing_yards': ['250'],  # Should become int
        })

        df_enforced = _enforce_dtypes(df)

        # Check dtypes
        self.assertEqual(df_enforced['player_id'].dtype, object)
        self.assertTrue(pd.api.types.is_integer_dtype(df_enforced['season']))
        self.assertTrue(pd.api.types.is_integer_dtype(df_enforced['week']))
        self.assertTrue(pd.api.types.is_integer_dtype(df_enforced['passing_yards']))


class TestWeekGuard(unittest.TestCase):
    """Test week guard prevents fetching future weeks"""

    def test_latest_completed_week_not_future(self):
        """Test that latest completed week is not in the future"""
        latest_week = get_latest_completed_week_2025()

        # Should be between 0-18
        self.assertGreaterEqual(latest_week, 0)
        self.assertLessEqual(latest_week, 18)

    @patch('tank01_client_robust.get_latest_completed_week_2025')
    def test_load_future_week_raises(self, mock_get_week):
        """Test that loading future week raises error"""
        # Mock latest completed week as 5
        mock_get_week.return_value = 5

        # Trying to load week 10 should raise
        with self.assertRaises(ValueError) as context:
            load_2025_week(10)

        self.assertIn("has not completed yet", str(context.exception))


class TestHostFailover(unittest.TestCase):
    """Test host failover logic"""

    @patch('tank01_client_robust.requests.Session')
    def test_host_failover_on_5xx(self, mock_session_class):
        """Test that host switches after consistent 5xx errors"""
        client = RobustTank01Client()

        mock_session = MagicMock()
        client.session = mock_session

        # First 3 attempts on primary host: 503
        mock_response_503 = Mock()
        mock_response_503.status_code = 503
        mock_response_503.raise_for_status.side_effect = Exception("503")

        # After failover, success on alternate host
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {'body': {'test': 'data'}}
        mock_response_success.raise_for_status.return_value = None

        mock_session.get.side_effect = [
            mock_response_503,  # Attempt 1: primary fails
            mock_response_503,  # Attempt 2: primary fails
            mock_response_503,  # Attempt 3: primary fails, triggers failover
            mock_response_success  # Attempt 4: alternate host succeeds
        ]

        initial_host = client.active_host

        # Make request
        result = client.fetch_tank01('/test', {'param': 'value'})

        # Verify host was switched
        self.assertNotEqual(client.active_host, initial_host)

        # Verify we got successful response
        self.assertEqual(result, {'body': {'test': 'data'}})


class TestNormalization(unittest.TestCase):
    """Test parameter normalization"""

    def test_season_type_normalization(self):
        """Test that seasonType is normalized to lowercase"""
        client = RobustTank01Client()

        params = {'seasonType': 'REG', 'week': '5'}
        normalized = client._normalize_params(params)

        self.assertEqual(normalized['seasonType'], 'reg')

    def test_invalid_season_type_raises(self):
        """Test that invalid seasonType raises error"""
        client = RobustTank01Client()

        params = {'seasonType': 'invalid', 'week': '5'}

        with self.assertRaises(AssertionError):
            client._normalize_params(params)

    def test_week_range_validation(self):
        """Test that week range is validated"""
        client = RobustTank01Client()

        # Valid week
        params = {'seasonType': 'reg', 'week': '10'}
        normalized = client._normalize_params(params)
        self.assertEqual(normalized['week'], '10')

        # Invalid week (too high)
        params_invalid = {'seasonType': 'reg', 'week': '25'}
        with self.assertRaises(AssertionError):
            client._normalize_params(params_invalid)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
