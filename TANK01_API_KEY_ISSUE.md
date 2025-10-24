# TANK01 API Key Issue and Resolution

## Critical Issue Discovered

**Status**: ⚠️ **TANK01 API KEY INVALID OR EXPIRED**

During testing, all TANK01 API endpoints returned **HTTP 403 Forbidden** errors, indicating the API key is invalid, expired, or lacks the necessary permissions.

---

## Test Results

Tested the following TANK01 endpoints with the configured API key:

| Endpoint | Result | Error |
|----------|--------|-------|
| `/getNFLNews` | ❌ FAILED | 403 Forbidden |
| `/getNFLGamesForWeek` | ❌ FAILED | 403 Forbidden |
| `/getNFLBoxScore` | ❌ FAILED | 403 Forbidden |
| `/getNFLTeams` | ❌ FAILED | 403 Forbidden |
| `/getNFLScoreboard` | ❌ FAILED | 403 Forbidden |

**Conclusion**: The API key `9c58e7cda2msh95af7473755205ep12f820jsn13cda64975cf` does not have access to any TANK01 endpoints.

---

## Root Causes

The 403 Forbidden error typically indicates one of the following:

1. **API Key Expired**: The RapidAPI subscription may have expired
2. **Invalid API Key**: The key may have been revoked or is incorrect
3. **Insufficient Permissions**: The subscription tier doesn't include these endpoints
4. **Rate Limit Exceeded**: Account may be blocked due to excessive usage
5. **Account Issue**: The RapidAPI account may have billing or verification issues

---

## Immediate Resolution - Graceful Fallback

✅ **The system has been updated to gracefully handle this issue:**

### Updates Made

#### 1. **tank01_stats_client.py**
- Added `Tank01APIAccessError` exception class
- Updated `_api_call_with_retry()` to detect 403 errors immediately
- Raises clear exception with actionable error message:
  ```python
  Tank01APIAccessError: TANK01 API Access Denied (403 Forbidden).
  The API key may be invalid, expired, or lack permissions.
  Please verify the TANK_API_KEY in phase1_config.py or contact RapidAPI support.
  ```

#### 2. **nfl_data_utils.py**
- Updated `load_current_season_data()` with graceful fallback:
  ```python
  Try TANK01 → If 403 error → Fall back to nfl_data_py → If that fails → Clear error message
  ```
- System will attempt nfl_data_py as fallback for 2025 season
- Provides clear error messages at each step

### Behavior Now

```
1. Detect season = 2025
2. Attempt TANK01 API
3. Get 403 Forbidden
4. Log warning: "⚠️  TANK01 API UNAVAILABLE"
5. Fall back to nfl_data_py for 2025
6. If nfl_data_py also doesn't have 2025:
   → Return clear error explaining both sources failed
```

---

## Long-Term Resolution - Renew API Key

To fully resolve the TANK01 integration, the API key needs to be renewed or replaced.

### Option 1: Renew Existing RapidAPI Subscription

1. **Visit RapidAPI**:
   - Go to https://rapidapi.com/
   - Log in to the account associated with the key

2. **Navigate to Tank01 NFL API**:
   - Go to https://rapidapi.com/tank01/api/tank01-nfl-live-in-game-real-time-statistics-nfl

3. **Check Subscription Status**:
   - Click on "Pricing" tab
   - Verify current subscription status
   - Check if subscription expired or needs payment

4. **Renew/Upgrade**:
   - Subscribe to appropriate tier
   - Required endpoints: `/getNFLBoxScore`, `/getNFLGamesForWeek`
   - Recommended tier: At least "Basic" or higher

5. **Get New API Key**:
   - Go to "Endpoints" tab
   - Find your API key in the code snippets
   - Copy the `x-rapidapi-key` value

6. **Update Configuration**:
   ```python
   # In phase1_config.py
   TANK_API_KEY = "your_new_api_key_here"
   ```

### Option 2: Use Different Account

1. Create new RapidAPI account
2. Subscribe to Tank01 NFL API
3. Get API key
4. Update `phase1_config.py`

### Option 3: Use Alternative Data Source

If TANK01 is not needed:
1. System will automatically use nfl_data_py as fallback
2. Note: nfl_data_py may not have 2025 data immediately
3. For 2025 season, wait for nfl_data_py to publish data (usually mid-season)

---

## Testing the Fix

After renewing the API key, test with:

```bash
# Test API access
python test_tank01_endpoints.py

# Expected output if fixed:
# ✅ SUCCESS - getNFLGamesForWeek endpoint is accessible
# ✅ SUCCESS - getNFLBoxScore endpoint is accessible
```

---

## Current System Behavior

### ✅ What Works Now

1. **Graceful Degradation**: System doesn't crash when TANK01 is unavailable
2. **Clear Error Messages**: Users see exactly what failed and why
3. **Automatic Fallback**: Attempts nfl_data_py as backup
4. **Historical Data**: All historical seasons (≤2024) work via nfl_data_py

### ⚠️ What's Limited

1. **2025 Data**: May not be available from either source until API key renewed
2. **Live Updates**: Cannot get real-time 2025 stats without TANK01
3. **Hybrid Loading**: Falls back to single source (nfl_data_py) only

---

## Error Messages You Might See

### During Data Loading

```
⚠️  TANK01 API UNAVAILABLE
   TANK01 API Access Denied (403 Forbidden). The API key may be invalid, expired,
   or lack permissions for endpoint '/getNFLGamesForWeek'. Please verify the
   TANK_API_KEY in phase1_config.py or contact RapidAPI support.

   🔄 Falling back to nfl_data_py for season 2025
   Note: Data may not be available or may be incomplete.
```

###  If Both Sources Fail

```
❌ ERROR: Neither TANK01 nor nfl_data_py have data for 2025
   TANK01 error: API access denied
   nfl_data_py error: HTTP Error 404: Not Found

ValueError: No data source available for 2025 season. TANK01 API key may need
renewal, and nfl_data_py doesn't have 2025 data yet.
```

---

## Recommended Actions

### Immediate (for 2025 season):

1. ✅ **Continue using system**: It will work with available historical data
2. ⚠️ **Renew TANK01 API key**: To enable 2025 data fetching
3. ℹ️ **Monitor nfl_data_py**: May publish 2025 data as season progresses

### Short-term (1-2 weeks):

1. Check if nfl_data_py has published 2025 data
2. If yes, system will automatically use it
3. If no, renew TANK01 key for real-time data

### Long-term:

1. Set up API key monitoring/alerts
2. Configure automatic renewals for critical APIs
3. Consider backup data sources

---

## FAQ

**Q: Will the system crash without a valid TANK01 key?**
A: No. The system now gracefully falls back to nfl_data_py.

**Q: Can I use the system with 2024 data only?**
A: Yes. All historical data works perfectly via nfl_data_py.

**Q: How much does TANK01 API cost?**
A: Visit https://rapidapi.com/tank01/api/tank01-nfl-live-in-game-real-time-statistics-nfl/pricing
   - Basic plans start around $10-20/month
   - Check which tier includes `/getNFLBoxScore` and `/getNFLGamesForWeek`

**Q: Is there a free alternative?**
A: Yes - nfl_data_py is free but may have delayed 2025 data publication.

**Q: When will nfl_data_py have 2025 data?**
A: Typically published mid-season or after season completion. Check https://github.com/cooperdff/nfl_data_py

---

## Files Modified for Fallback

| File | Change | Purpose |
|------|--------|---------|
| `tank01_stats_client.py` | Added exception handling | Detect 403 errors immediately |
| `nfl_data_utils.py` | Added fallback logic | Use nfl_data_py when TANK01 fails |
| `TANK01_API_KEY_ISSUE.md` | Created documentation | Explain issue and resolution |

---

## Summary

**Current Status**: ✅ System is stable with fallback logic

**To Fully Restore TANK01 Integration**:
1. Renew RapidAPI subscription
2. Update API key in `phase1_config.py`
3. Test with `python test_tank01_endpoints.py`
4. System will automatically use TANK01 for 2025 data

**Alternative**: Wait for nfl_data_py to publish 2025 data (system will auto-detect)

---

**Last Updated**: October 24, 2025
**Status**: Waiting for API key renewal or 2025 data publication
