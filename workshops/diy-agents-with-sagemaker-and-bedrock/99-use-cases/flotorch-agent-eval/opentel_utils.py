from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from queue import Queue
from typing import Sequence

# Create a custom exporter that stores spans in a queue
class QueueSpanExporter(SpanExporter):
    def __init__(self):
        self.spans = Queue()
        
    def export(self, spans: Sequence[ReadableSpan]) -> None:
        for span in spans:
            self.spans.put(span)
        return None
    
    def clear(self) -> None:
        with self.spans.mutex:
            self.spans.queue.clear()
    
    def force_flush(self, timeout_millis: float = 30000) -> None:
        return None
    
    def shutdown(self) -> None:
        return None