import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VLM_BUILD_ID = "vlm-cpu-inference-v4"
WEB_BUILD_ID = "auto-pdf-text-v7"


def wait_for(url: str, timeout: int = 30) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def service_version(url: str):
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            health = json.loads(response.read().decode("utf-8"))
            return health.get("version")
    except (OSError, ValueError):
        return None


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    services = [
        (
            "VLM",
            "http://127.0.0.1:8001/health",
            ROOT / "VLM" / "app.py",
            VLM_BUILD_ID,
        ),
        (
            "Web app",
            "http://127.0.0.1:5000/health",
            ROOT / "Web" / "app.py",
            WEB_BUILD_ID,
        ),
    ]
    started = []

    for name, health_url, script, expected_version in services:
        if wait_for(health_url, timeout=1):
            version = service_version(health_url)
            if version != expected_version:
                print(
                    f"An older {name} is already running "
                    f"(version: {version or 'unknown'}). Stop the existing "
                    "service, then run this launcher again."
                )
                for _, child in started:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=10)
                return 1
            print(f"{name} is already running at {health_url.rsplit('/health', 1)[0]}")
            continue

        print(f"Starting {name}...")
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            env=env,
        )
        started.append((name, process))

        if wait_for(health_url, timeout=30):
            print(f"{name} is ready at {health_url.rsplit('/health', 1)[0]}")
            continue

        exit_code = process.poll()
        if exit_code is None:
            reason = "did not become healthy within 30 seconds"
        else:
            reason = f"exited with code {exit_code}"
        print(f"{name} {reason}. Check the server output above.")
        for _, child in started:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
        return 1

    print("\nPress Ctrl+C to stop services started by this launcher.")
    try:
        while True:
            for name, process in started:
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"{name} stopped unexpectedly with exit code {exit_code}.")
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
    finally:
        for _, process in started:
            if process.poll() is None:
                process.terminate()
        for _, process in started:
            if process.poll() is None:
                process.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
