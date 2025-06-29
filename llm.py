import asyncio
import httpx


class LLMClient:
    def __init__(self, system_prompt: str, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.messages = [{"role": "system", "content": system_prompt}]

    async def call_stream(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "phi4",
            "messages": self.messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream('POST', f"{self.base_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                full_response = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    full_response += content
                                    yield full_response
                        except json.JSONDecodeError:
                            continue

                self.messages.append({"role": "assistant", "content": full_response})
