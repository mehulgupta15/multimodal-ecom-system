import asyncio
import time
import httpx

# The URL where your search engine backend is running
TARGET_URL = "http://127.0.0.1:8000/search" 

# Define the "Curveball Queries"
TEST_CASES = [
    {"name": "Normal Word", "query": "running shoes"},
    {"name": "Empty Input", "query": ""},
    {"name": "Gibberish", "query": "asdfghjkl12345!@#$%"},
    {"name": "Extremely Long Description", "query": "red " * 500},  # 500 repetitions
]

# Simulate a crowd by duplicating the test cases to hit the server at once
CONCURRENT_USERS = 5
TRACKED_QUERIES = TEST_CASES * CONCURRENT_USERS

async def send_request(client, case, metrics):
    # FIXED: Send payload via form-encoded data field key matching text_query validation logic
    form_data = {"text_query": case["query"]}
    start_time = time.perf_counter()
    
    try:
        # FIXED: Using data= parameters instead of json= blocks to comply with Form endpoints
        response = await client.post(TARGET_URL, data=form_data, timeout=10.0)
        duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Check if the server handled it gracefully (200 OK or a proper 400 Bad Request error)
        if response.status_code in [200, 400]:
            metrics["success"] += 1
        else:
            metrics["failed"] += 1
            
        metrics["latencies"].append(duration)
        print(f"[{case['name']}] Status: {response.status_code} | Time: {duration:.2f}ms")

    except Exception as e:
        duration = (time.perf_counter() - start_time) * 1000
        metrics["failed"] += 1
        metrics["latencies"].append(duration)
        print(f"[{case['name']}] Request Failed: {e} | Time: {duration:.2f}ms")

async def main():
    metrics = {"success": 0, "failed": 0, "latencies": []}
    
    print(f"🚀 Starting Day 10 Test: Firing {len(TRACKED_QUERIES)} simultaneous requests...\n")
    test_start = time.perf_counter()

    # Open a single client session for maximum speed
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, case, metrics) for case in TRACKED_QUERIES]
        await asyncio.gather(*tasks)

    test_end = time.perf_counter()
    total_time = (test_end - test_start) * 1000

    # --- The Performance Scorecard ---
    print("\n" + "="*40)
    print("       PERFORMANCE SCORECARD")
    print("="*40)
    print(f"Successful Requests (Graceful): {metrics['success']}")
    print(f"Broken Requests (Crashed):     {metrics['failed']}")
    
    if metrics["latencies"]:
        avg_speed = sum(metrics["latencies"]) / len(metrics["latencies"])
        max_speed = max(metrics["latencies"])
        print(f"Average Response Speed:        {avg_speed:.2f} ms")
        print(f"Slowest Response Speed:        {max_speed:.2f} ms")
    
    print(f"Total Test Duration:           {total_time:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())