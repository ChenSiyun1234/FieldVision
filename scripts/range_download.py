"""Download a large file with parallel HTTP range requests.

    python scripts/range_download.py URL OUT [--workers 4] [--chunk-mb 64]

The NASS server answers a plain GET for the Crop Sequence Boundaries archive with an empty body
but serves byte ranges. Chunks are written into a preallocated file and recorded in OUT.done, so
an interrupted download resumes; every chunk must come back as HTTP 206 with the exact length.
"""
import argparse
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('url')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk-mb', type=int, default=64)
    a = ap.parse_args(argv)
    chunk = a.chunk_mb << 20
    head = urllib.request.urlopen(urllib.request.Request(a.url, method='HEAD'), timeout=60).headers
    size = int(head['Content-Length'])
    print('size', size, 'last-modified', head.get('Last-Modified'), flush=True)
    done_path = a.out + '.done'
    done = {int(x) for x in open(done_path).read().split()} if os.path.exists(done_path) else set()
    if not os.path.exists(a.out) or os.path.getsize(a.out) != size:
        done = set()
        with open(a.out, 'wb') as f:
            f.truncate(size)
    lock = threading.Lock()

    def get(start):
        end = min(start + chunk, size) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(a.url, headers={'Range': f'bytes={start}-{end}'})
                r = urllib.request.urlopen(req, timeout=120)
                if r.status != 206:
                    raise IOError(f'HTTP {r.status} instead of 206')
                data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f'got {len(data)} bytes, expected {end - start + 1}')
                with lock:
                    with open(a.out, 'r+b') as f:
                        f.seek(start)
                        f.write(data)
                    with open(done_path, 'a') as f:
                        f.write(f'{start}\n')
                return
            except Exception as e:
                print('retry', start, attempt, repr(e), flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f'chunk at {start} failed 8 times')

    todo = [s for s in range(0, size, chunk) if s not in done]
    t0 = time.time()
    print('chunks to fetch', len(todo), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(get, todo), 1):
            if i % 5 == 0 or i == len(todo):
                print(f'{i}/{len(todo)} chunks, {time.time() - t0:.0f} s', flush=True)
    os.remove(done_path)
    print('complete', size, 'bytes', flush=True)


if __name__ == '__main__':
    main()
