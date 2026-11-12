"""跨机执行：通过 SSH 在远端跑命令、传文件、读日志。

控制面不假设自己部署在集群里——它可以是一台运维机，只要有到各节点的 SSH 通道。
这也是"迁移"能跨机器发生的前提：状态文件必须能从源机取回、再送到目标机。
"""

from __future__ import annotations

import posixpath
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import paramiko


@dataclass
class HostSpec:
    name: str
    host: str
    port: int
    user: str = "root"
    password: str = ""
    gpu: int | None = None
    label: str = ""
    worker_port: int | None = None

    @property
    def display(self) -> str:
        return self.label or self.name

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "HostSpec":
        return cls(
            name=name,
            host=str(data["host"]),
            port=int(data.get("port", 22)),
            user=str(data.get("user", "root")),
            password=str(data.get("password", "")),
            gpu=(int(data["gpu"]) if data.get("gpu") is not None else None),
            label=str(data.get("label", "") or name),
            worker_port=(int(data["worker_port"]) if data.get("worker_port") is not None else None),
        )


class RemoteHost:
    """一个远端节点的句柄。连接懒建立，失败重试一次。"""

    @property
    def display(self) -> str:
        return self.spec.label or self.spec.name

    def __init__(self, spec: HostSpec, timeout: float = 20.0):
        self.spec = spec
        self.timeout = timeout
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    # ---------- 连接 ----------

    def _connect(self) -> paramiko.SSHClient:
        if self._client is not None:
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                return self._client
            self.close()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.spec.host,
            port=self.spec.port,
            username=self.spec.user,
            password=self.spec.password,
            timeout=self.timeout,
            banner_timeout=self.timeout,
            auth_timeout=self.timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        self._client = client
        return client

    def close(self) -> None:
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:  # noqa: BLE001
                pass
            self._sftp = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def __enter__(self) -> "RemoteHost":
        self._connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def sftp(self) -> paramiko.SFTPClient:
        self._connect()
        if self._sftp is None:
            self._sftp = self._client.open_sftp()  # type: ignore[union-attr]
        return self._sftp

    # ---------- 命令 ----------

    def run(self, cmd: str, timeout: float = 60.0) -> tuple[int, str, str]:
        client = self._connect()
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    def run_detached(self, cmd: str, read_timeout: float = 5.0) -> str:
        """执行一条"自己后台化"的命令。

        ``nohup ... &`` 的子孙进程可能继承 SSH 通道的 fd，导致 ``exec_command``
        一直等 EOF。这里显式读一小段时间就关闭通道，不阻塞控制器。
        """

        client = self._connect()
        transport = client.get_transport()
        if transport is None:
            return ""
        chan = transport.open_session()
        chan.settimeout(read_timeout)
        chan.exec_command(cmd)
        buf = b""
        deadline = time.time() + read_timeout
        try:
            while time.time() < deadline:
                if chan.recv_ready():
                    buf += chan.recv(4096)
                elif chan.exit_status_ready():
                    break
                else:
                    time.sleep(0.05)
        except Exception:  # noqa: BLE001
            pass
        try:
            chan.close()
        except Exception:  # noqa: BLE001
            pass
        return buf.decode("utf-8", "replace")

    def run_ok(self, cmd: str, timeout: float = 60.0) -> str:
        rc, out, err = self.run(cmd, timeout)
        if rc != 0:
            raise RuntimeError(f"[{self.spec.name}] 命令失败 rc={rc}: {cmd}\n{err[-800:]}")
        return out

    # ---------- 文件 ----------

    def exists(self, remote: str) -> bool:
        try:
            self.sftp.stat(remote)
            return True
        except IOError:
            return False

    def read_text(self, remote: str, limit: int | None = None) -> str:
        try:
            with self.sftp.open(remote, "r") as fh:
                data = fh.read()
        except IOError:
            return ""
        text = data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
        return text[-limit:] if limit else text

    def write_text(self, remote: str, content: str) -> None:
        self.makedirs(posixpath.dirname(remote))
        tmp = remote + ".tmp"
        with self.sftp.open(tmp, "w") as fh:
            fh.write(content.encode("utf-8"))
        self.run(f"mv -f {_q(tmp)} {_q(remote)}")

    def put(self, local: str | Path, remote: str) -> None:
        self.makedirs(posixpath.dirname(remote))
        self.sftp.put(str(local), remote)

    def get(self, remote: str, local: str | Path) -> bool:
        try:
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            self.sftp.get(remote, str(local))
            return True
        except IOError:
            return False

    def makedirs(self, remote: str) -> None:
        if remote:
            self.run(f"mkdir -p {_q(remote)}")

    def rm(self, remote: str) -> None:
        self.run(f"rm -rf {_q(remote)}")

    def listdir(self, remote: str) -> list[str]:
        try:
            return sorted(self.sftp.listdir(remote))
        except IOError:
            return []

    # ---------- 进程 ----------

    def pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        rc, out, _ = self.run(f"kill -0 {pid} 2>/dev/null && echo alive || echo dead")
        return "alive" in out

    def kill_tree(self, pid: int) -> None:
        """杀掉进程及其子进程（推理服务只有一个进程，这里用 start_new_session 的进程组）。"""

        if pid <= 0:
            return
        self.run(
            f"kill -TERM -{pid} 2>/dev/null; sleep 1; kill -KILL -{pid} 2>/dev/null; "
            f"kill -KILL {pid} 2>/dev/null; true"
        )

    def tail(self, remote: str, lines: int = 40) -> str:
        if not self.exists(remote):
            return ""
        rc, out, _ = self.run(f"tail -n {lines} {_q(remote)}")
        return out if rc == 0 else ""

    def http_status(self, port: int, path: str = "/health") -> int:
        rc, out, _ = self.run(
            f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 3 http://127.0.0.1:{port}{path}"
        )
        try:
            return int(out.strip() or 0)
        except ValueError:
            return 0

    def gpu_snapshot(self) -> str:
        rc, out, _ = self.run(
            "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used "
            "--format=csv,noheader"
        )
        return out.strip() if rc == 0 else ""


def _q(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"


def wait_until(predicate, timeout_s: float, interval_s: float = 0.5) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def parse_hosts(path: str | Path) -> dict[str, HostSpec]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    hosts: dict[str, HostSpec] = {}
    for name, spec in (data.get("hosts") or {}).items():
        hosts[str(name)] = HostSpec.from_dict(str(name), spec)
    return hosts


def iter_names(specs: dict[str, HostSpec]) -> Iterable[str]:
    return specs.keys()
