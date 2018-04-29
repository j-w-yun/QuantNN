import time
import requests


def http_get(url, params=None, timeout=120, max_retries=3):
    r = None
    for retries in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if not r.ok:
                time.sleep(3 * (retries + 1))
                continue
        except:
            time.sleep(3 * (retries + 1))
            pass

    if r is None:
        return None

    return r.content
