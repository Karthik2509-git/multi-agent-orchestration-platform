"""Tests for SafeHTTPGetTool including SSRF, DNS-rebinding, and size protections."""

from unittest.mock import AsyncMock, patch

import pytest

from src.app.tools.http_tool import SafeHTTPGetTool


@pytest.fixture
def http_tool() -> SafeHTTPGetTool:
    return SafeHTTPGetTool(
        allowed_domains=["httpbin.org", "api.github.com"],
        timeout=2.0,
        max_size_bytes=100,
    )


@pytest.mark.asyncio
async def test_blocked_domain_rejected(http_tool: SafeHTTPGetTool) -> None:
    """Test that domains not in the allowlist are rejected."""
    result = await http_tool.execute(url="https://evil.com/data")
    assert result.success is False
    assert "not in the allowed domains list" in result.error


@pytest.mark.asyncio
async def test_private_and_localhost_ip_rejected(http_tool: SafeHTTPGetTool) -> None:
    """Test that direct loopback and private IP addresses are rejected."""
    res1 = await http_tool.execute(url="http://127.0.0.1:8000/health")
    assert res1.success is False
    assert "Direct IP addresses" in res1.error or "not in the allowed" in res1.error

    res2 = await http_tool.execute(url="http://192.168.1.1/admin")
    assert res2.success is False

    res3 = await http_tool.execute(url="http://localhost:8000/health")
    assert res3.success is False
    assert "not in the allowed domains list" in res3.error


@pytest.mark.asyncio
async def test_dns_rebinding_ssrf_protection(http_tool: SafeHTTPGetTool) -> None:
    """Test that a domain resolving to a private/loopback IP is blocked."""
    # Simulate httpbin.org resolving to a loopback address (127.0.0.1)
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 443))]
        result = await http_tool.execute(url="https://httpbin.org/get")
        assert result.success is False
        assert "SSRF Security Violation" in result.error
        assert "Forbidden IP address range" in result.error


@pytest.mark.asyncio
async def test_dns_rebinding_private_range_protection(http_tool: SafeHTTPGetTool) -> None:
    """Test that a domain resolving to a 10.x.x.x private address is blocked."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("10.0.1.25", 443))]
        result = await http_tool.execute(url="https://httpbin.org/get")
        assert result.success is False
        assert "SSRF Security Violation" in result.error


@pytest.mark.asyncio
async def test_allowed_domain_mock_success(http_tool: SafeHTTPGetTool) -> None:
    """Test successful GET request to an allowed domain with mocked HTTP response."""
    with patch("socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]

        with patch("httpx.AsyncClient.stream") as mock_stream:
            # Setup mock response stream
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.url = "https://httpbin.org/get"
            mock_resp.headers = {"content-type": "application/json"}

            async def mock_aiter():
                yield b'{"status": "ok"}'

            mock_resp.aiter_bytes = mock_aiter
            mock_stream.return_value.__aenter__.return_value = mock_resp

            result = await http_tool.execute(url="https://httpbin.org/get")
            assert result.success is True
            assert result.data["status_code"] == 200
            assert '{"status": "ok"}' in result.data["content"]
            assert result.data["truncated"] is False


@pytest.mark.asyncio
async def test_response_size_truncation(http_tool: SafeHTTPGetTool) -> None:
    """Test that responses exceeding max_size_bytes are truncated."""
    with patch("socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]

        with patch("httpx.AsyncClient.stream") as mock_stream:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.url = "https://httpbin.org/large"
            mock_resp.headers = {}

            # Generate 200 bytes while max_size_bytes is 100
            async def mock_aiter():
                yield b"A" * 70
                yield b"B" * 70
                yield b"C" * 60

            mock_resp.aiter_bytes = mock_aiter
            mock_stream.return_value.__aenter__.return_value = mock_resp

            result = await http_tool.execute(url="https://httpbin.org/large")
            assert result.success is True
            assert result.data["truncated"] is True
            assert result.data["bytes_read"] <= 100


@pytest.mark.asyncio
async def test_redirect_not_followed(http_tool: SafeHTTPGetTool) -> None:
    """Test that HTTP redirects return status code and do not automatically follow."""
    with patch("socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]

        with patch("httpx.AsyncClient.stream") as mock_stream:
            mock_resp = AsyncMock()
            mock_resp.status_code = 302
            mock_resp.url = "https://httpbin.org/redirect"
            mock_resp.headers = {"location": "https://other.com"}

            async def mock_aiter():
                yield b""

            mock_resp.aiter_bytes = mock_aiter
            mock_stream.return_value.__aenter__.return_value = mock_resp

            result = await http_tool.execute(url="https://httpbin.org/redirect")
            assert result.success is True
            assert result.data["status_code"] == 302
            assert "Redirect" in result.data["note"]
