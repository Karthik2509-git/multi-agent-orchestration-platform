"""Safe HTTP GET tool with DNS-rebinding, SSRF, and response size protections."""

import ipaddress
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from src.app.core.logging import get_logger
from src.app.tools.base import BaseTool, ToolResult

logger = get_logger(__name__)


class SafeHTTPGetTool(BaseTool):
    """Safely executes HTTP GET requests against explicitly allowlisted domains."""

    name: str = "http_get"
    description: str = (
        "Performs a safe HTTP GET request to a permitted domain. "
        "Guards against SSRF, DNS-rebinding, and redirects. "
        "Example: 'https://httpbin.org/get' or 'https://api.github.com/zen'."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The absolute HTTP or HTTPS URL to fetch.",
            }
        },
        "required": ["url"],
    }

    def __init__(
        self,
        allowed_domains: List[str],
        timeout: float = 10.0,
        max_size_bytes: int = 100_000,
    ):
        self.allowed_domains = [d.lower().strip() for d in allowed_domains if d.strip()]
        self.timeout = timeout
        self.max_size_bytes = max_size_bytes

    def _is_domain_allowed(self, hostname: str) -> bool:
        """Verify that the target hostname is present in the domain allowlist."""
        normalized_host = hostname.lower()
        for allowed in self.allowed_domains:
            if normalized_host == allowed or normalized_host.endswith("." + allowed):
                return True
        return False

    def _validate_ip(self, ip_str: str) -> Optional[str]:
        """Check if an IP address belongs to private, loopback, or reserved ranges."""
        try:
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return f"Forbidden IP address range: {ip_str}"
        except ValueError:
            return f"Invalid IP address format: {ip_str}"
        return None

    def _validate_and_resolve_url(self, url: str) -> Optional[str]:
        """Validate URL scheme, domain allowlist, and resolve all DNS IP addresses.

        Returns an error string if invalid, or None if valid.
        """
        try:
            parsed = urlparse(url)
        except Exception as e:
            return f"Malformed URL: {str(e)}"

        if parsed.scheme not in ("http", "https"):
            return f"Invalid URL scheme '{parsed.scheme}'. Only 'http' and 'https' are permitted."

        hostname = parsed.hostname
        if not hostname:
            return "URL is missing a valid hostname"

        # Reject raw IP addresses in hostname unless explicitly listed in allowed domains
        try:
            ipaddress.ip_address(hostname)
            if hostname.lower() not in self.allowed_domains:
                return (
                    f"Direct IP addresses ({hostname}) are not permitted "
                    "unless explicitly allowlisted."
                )
        except ValueError:
            # Not a raw IP address, proceed with domain validation
            pass

        # Check domain allowlist
        if not self._is_domain_allowed(hostname):
            return f"Domain '{hostname}' is not in the allowed domains list: {self.allowed_domains}"

        # DNS-Rebinding Protection: Resolve hostname and inspect all target IPs
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            if not addr_info:
                return f"Could not resolve host '{hostname}'"

            for addr in addr_info:
                ip_str = addr[4][0]
                ip_error = self._validate_ip(ip_str)
                if ip_error:
                    return f"SSRF Security Violation: {ip_error} (resolved from {hostname})"
        except socket.gaierror as dns_err:
            return f"DNS resolution failed for host '{hostname}': {str(dns_err)}"

        return None

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a safe HTTP GET request with size and redirect protections."""
        url = kwargs.get("url", "")
        if not url or not isinstance(url, str):
            return ToolResult(
                success=False,
                error="Invalid input: 'url' must be a non-empty string",
            )

        url = url.strip()
        validation_error = self._validate_and_resolve_url(url)
        if validation_error:
            logger.warning("SafeHTTPGetTool rejected request to %s: %s", url, validation_error)
            return ToolResult(success=False, error=validation_error)

        logger.info("Executing Safe HTTP GET request to %s", url)

        try:
            # Disable redirects to prevent redirect-based SSRF attacks
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=self.timeout,
            ) as client:
                async with client.stream("GET", url) as response:
                    # Enforce body size limits to prevent LLM context exhaustion
                    content_chunks = []
                    bytes_read = 0
                    truncated = False

                    async for chunk in response.aiter_bytes():
                        bytes_read += len(chunk)
                        if bytes_read > self.max_size_bytes:
                            # Truncate content
                            remaining = self.max_size_bytes - (bytes_read - len(chunk))
                            if remaining > 0:
                                content_chunks.append(chunk[:remaining])
                            truncated = True
                            break
                        content_chunks.append(chunk)

                    raw_body = b"".join(content_chunks)
                    body_text = raw_body.decode("utf-8", errors="replace")

                    result_data = {
                        "url": str(response.url),
                        "status_code": response.status_code,
                        "truncated": truncated,
                        "bytes_read": len(raw_body),
                        "content": body_text,
                    }

                    if 300 <= response.status_code < 400:
                        location = response.headers.get("location", "")
                        result_data["note"] = (
                            f"Redirect to '{location}' was not followed for security reasons."
                        )

                    return ToolResult(success=True, data=result_data)

        except httpx.TimeoutException:
            logger.warning("HTTP request to %s timed out after %ss", url, self.timeout)
            return ToolResult(
                success=False,
                error=f"HTTP request timed out after {self.timeout} seconds",
            )
        except httpx.RequestError as req_err:
            logger.warning("HTTP request to %s failed: %s", url, str(req_err))
            return ToolResult(
                success=False,
                error=f"HTTP connection error: {str(req_err)}",
            )
        except Exception as ex:
            logger.error("Unexpected error in SafeHTTPGetTool for %s: %s", url, str(ex))
            return ToolResult(success=False, error=f"Unexpected HTTP tool error: {str(ex)}")
