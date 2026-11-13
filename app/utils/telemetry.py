
import time


class Telemetry:

    def __init__(self):

        self.start_time = time.perf_counter()

        self.retrieval_latency_ms = 0.0
        self.rerank_latency_ms = 0.0
        self.generation_latency_ms = 0.0

    def calculate_total_latency(self):

        return round(
            (time.perf_counter() - self.start_time) * 1000,
            2,
        )

    def get_metadata(self):

        return {
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "rerank_latency_ms": self.rerank_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.calculate_total_latency(),
        }
