from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

from request_limiter import FairRequestLimiter, RequestLimitExceeded


class RequestLimiterTests(unittest.TestCase):
    def test_request_rate_limit_is_shared_between_sessions(self):
        limiter = FairRequestLimiter(
            max_concurrent=1,
            max_requests_per_minute=1,
            max_requests_per_session_per_minute=1,
        )

        with limiter.request_slot("session-a"):
            pass

        with self.assertRaisesRegex(RequestLimitExceeded, "many requests"):
            with limiter.request_slot("session-b"):
                pass

    def test_concurrent_requests_stay_within_service_limit(self):
        user_count = 8
        limiter = FairRequestLimiter(
            max_concurrent=2,
            max_requests_per_minute=20,
            max_requests_per_session_per_minute=5,
            wait_timeout_seconds=3,
        )
        start_together = threading.Barrier(user_count)
        active = 0
        peak_active = 0
        lock = threading.Lock()

        def make_request(index):
            nonlocal active, peak_active
            start_together.wait(timeout=3)
            with limiter.request_slot(f"session-{index}"):
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                time.sleep(0.02)
                with lock:
                    active -= 1

        with ThreadPoolExecutor(max_workers=user_count) as executor:
            list(executor.map(make_request, range(user_count)))

        self.assertEqual(peak_active, 2)


if __name__ == "__main__":
    unittest.main()
