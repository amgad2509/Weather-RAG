import json

def sse_event(data: dict, event: str = "message") -> str:
    # SSE format: "event: x\ndata: {...}\n\n"
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
