"""跨机执行与迁移支撑。"""

from .ssh import HostSpec, RemoteHost, parse_hosts  # noqa: F401

__all__ = ["HostSpec", "RemoteHost", "parse_hosts"]
