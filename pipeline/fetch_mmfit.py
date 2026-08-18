"""
Selectively extract MM-Fit IMU + label files from the remote zip using HTTP range
requests. Downloading the full 1.74 GB archive and unpacking it is not viable on a
disk with <5 GB free, and ~90% of the archive is pose/skeleton data we do not use.

Reads the zip central directory over the network, then fetches only the members we
need. Works because S3 honours Range requests.
"""
import io
import os
import struct
import sys
import urllib.request
import zipfile

URL = "https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip"


def fetch_range(url, start, end):
    """Fetch bytes [start, end] inclusive."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def content_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def read_central_directory(url):
    """Locate and parse the zip End Of Central Directory + central directory."""
    size = content_length(url)
    tail = fetch_range(url, max(0, size - 65_600), size - 1)

    eocd = tail.rfind(b"PK\x05\x06")
    if eocd == -1:
        raise RuntimeError("EOCD signature not found")

    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12 : eocd + 20])

    # Zip64 fallback: 0xFFFFFFFF sentinels mean the real values live in the
    # zip64 EOCD record. MM-Fit is under 4 GB so this is defensive only.
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        loc = tail.rfind(b"PK\x06\x07")
        if loc == -1:
            raise RuntimeError("zip64 locator not found")
        (z64_off,) = struct.unpack("<Q", tail[loc + 8 : loc + 16])
        z64 = fetch_range(url, z64_off, z64_off + 55)
        cd_size, cd_offset = struct.unpack("<QQ", z64[40:56])

    return size, cd_offset, cd_size


def parse_entries(cd_bytes):
    """Yield (filename, local_header_offset, compressed_size) per central dir entry."""
    pos = 0
    while pos < len(cd_bytes):
        if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
            break
        (comp_size, uncomp_size, name_len, extra_len, comment_len) = struct.unpack(
            "<II", cd_bytes[pos + 20 : pos + 28]
        ) + struct.unpack("<HHH", cd_bytes[pos + 28 : pos + 34])
        (local_off,) = struct.unpack("<I", cd_bytes[pos + 42 : pos + 46])
        name = cd_bytes[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        extra = cd_bytes[pos + 46 + name_len : pos + 46 + name_len + extra_len]

        # Resolve zip64 extended info if any size field is saturated.
        if 0xFFFFFFFF in (comp_size, uncomp_size, local_off):
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[e : e + 4])
                if hid == 0x0001:
                    vals = extra[e + 4 : e + 4 + hsz]
                    v, vi = [], 0
                    for orig in (uncomp_size, comp_size, local_off):
                        if orig == 0xFFFFFFFF and vi + 8 <= len(vals):
                            v.append(struct.unpack("<Q", vals[vi : vi + 8])[0])
                            vi += 8
                        else:
                            v.append(orig)
                    uncomp_size, comp_size, local_off = v[1], v[0], v[2]
                    # note: order above is (uncomp, comp, offset) in the record
                    break
                e += 4 + hsz

        yield name, local_off, comp_size
        pos += 46 + name_len + extra_len + comment_len


def wanted(name):
    """IMU streams from both smartwatches + earbud, plus labels. Skip pose/skeleton."""
    base = os.path.basename(name)
    if not base:
        return False
    if base.endswith("_labels.csv"):
        return True
    # e.g. w00_sw_l_acc.npy, w00_sw_r_gyr.npy, w00_eb_acc.npy
    return any(
        base.endswith(f"_{dev}_{sig}.npy")
        for dev in ("sw_l", "sw_r", "eb")
        for sig in ("acc", "gyr", "mag")
    )


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    os.makedirs(out_root, exist_ok=True)

    print("reading central directory...", flush=True)
    _, cd_offset, cd_size = read_central_directory(URL)
    cd = fetch_range(URL, cd_offset, cd_offset + cd_size - 1)

    entries = list(parse_entries(cd))
    print(f"  {len(entries)} members in archive")

    targets = [e for e in entries if wanted(e[0])]
    total = sum(e[2] for e in targets)
    print(f"  {len(targets)} match IMU/label filter, {total/1e6:.1f} MB compressed")

    if not targets:
        raise SystemExit("no matching members -- naming convention differs, inspect manually")

    for i, (name, local_off, comp_size) in enumerate(sorted(targets), 1):
        dest = os.path.join(out_root, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        # Local header is variable-length; over-fetch then let zipfile parse it.
        blob = fetch_range(URL, local_off, local_off + comp_size + 4096)
        lh_name_len, lh_extra_len = struct.unpack("<HH", blob[26:30])
        data_start = 30 + lh_name_len + lh_extra_len
        raw = blob[data_start : data_start + comp_size]

        (method,) = struct.unpack("<H", blob[8:10])
        if method == 0:
            data = raw
        elif method == 8:
            import zlib
            data = zlib.decompress(raw, -zlib.MAX_WBITS)
        else:
            raise RuntimeError(f"unsupported compression method {method} for {name}")

        with open(dest, "wb") as f:
            f.write(data)
        print(f"  [{i}/{len(targets)}] {name} -> {len(data)/1e6:.1f} MB", flush=True)

    print("done")


if __name__ == "__main__":
    main()
