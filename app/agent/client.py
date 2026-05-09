import httpx

from app.config import settings


class CentralServerClient:
    def __init__(self) -> None:
        self.base_url = settings.central_server_url.rstrip("/")
        self.api_key = settings.central_api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-shs-node-api-key": self.api_key,
        }

    async def register_node(self, payload: dict) -> dict:
        url = f"{self.base_url}/api/agent/register"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()