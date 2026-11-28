"""Calculate the HCU memory latency model in microseconds (decimal TB/s)."""

import argparse
import json
import math


# Project reference rates. A fixed-clock measurement may override either rate.
REFERENCE_TBPS = {"gfx936": (1.5, 6.0), "gfx938": (2.0, 6.0)}


def _whole(value, label, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be a whole number >= {minimum}")


def _time_us(byte_count, rate_tbps, level):
    if rate_tbps is None or not math.isfinite(rate_tbps) or rate_tbps <= 0:
        raise ValueError(f"{level} bandwidth must be finite and positive (decimal TB/s)")
    return byte_count / (rate_tbps * 1e6)


def gemm_floor(*, m, n, k, tile_m, tile_n, x_item_bytes, w_item_bytes,
               y_item_bytes, hbm_tbps, l2_tbps, extra_hbm_bytes=0, extra_l2_bytes=0):
    """Model a dense MxK by KxN GEMM with one output write and unsplit K.

    tile_m and tile_n are CTA extents. m_tiles and n_tiles are the resulting
    counts. Partial tiles access their valid rows and columns. Add bytes for
    scales, padding or other actual traffic at the appropriate memory level.
    """
    for name, value in (("m", m), ("n", n), ("k", k), ("tile_m", tile_m),
                        ("tile_n", tile_n), ("x_item_bytes", x_item_bytes),
                        ("w_item_bytes", w_item_bytes), ("y_item_bytes", y_item_bytes)):
        _whole(value, name, 1)
    _whole(extra_hbm_bytes, "extra_hbm_bytes", 0)
    _whole(extra_l2_bytes, "extra_l2_bytes", 0)
    m_tiles = (m + tile_m - 1) // tile_m
    n_tiles = (n + tile_n - 1) // tile_n
    x_bytes = m * k * x_item_bytes
    w_bytes = k * n * w_item_bytes
    y_bytes = m * n * y_item_bytes
    hbm_bytes = x_bytes + w_bytes + y_bytes + extra_hbm_bytes
    l2_bytes = (m_tiles - 1) * w_bytes + (n_tiles - 1) * x_bytes + extra_l2_bytes
    hbm_us = _time_us(hbm_bytes, hbm_tbps, "HBM")
    l2_us = _time_us(l2_bytes, l2_tbps, "L2")
    return {
        "m_tiles": m_tiles, "n_tiles": n_tiles, "ops": 2 * m * n * k,
        "x_bytes": x_bytes, "w_bytes": w_bytes, "y_bytes": y_bytes,
        "extra_hbm_bytes": extra_hbm_bytes, "extra_l2_bytes": extra_l2_bytes,
        "hbm_tbps": hbm_tbps, "l2_tbps": l2_tbps,
        "hbm_bytes": hbm_bytes, "l2_bytes": l2_bytes,
        "hbm_us": hbm_us, "l2_us": l2_us, "floor_us": hbm_us + l2_us,
    }


def stream_floor(*, read_bytes, write_bytes, hbm_tbps):
    """Model elementwise or reduction traffic; include every HBM pass."""
    _whole(read_bytes, "read_bytes", 0)
    _whole(write_bytes, "write_bytes", 0)
    byte_count = read_bytes + write_bytes
    if byte_count == 0:
        raise ValueError("read_bytes + write_bytes must be positive")
    hbm_us = _time_us(byte_count, hbm_tbps, "HBM")
    return {
        "read_bytes": read_bytes, "write_bytes": write_bytes, "hbm_tbps": hbm_tbps,
        "hbm_bytes": byte_count, "l2_bytes": 0,
        "hbm_us": hbm_us, "l2_us": 0.0, "floor_us": hbm_us,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operator", required=True)
    gemm = commands.add_parser("gemm", help="dense tiled GEMM")
    stream = commands.add_parser("stream", help="HBM elementwise or reduction traffic")
    for command in (gemm, stream):
        command.add_argument("--arch", required=True, help="gfx936, gfx938 or a measured target")
        command.add_argument("--hbm-tbps", type=float, help="measured HBM rate in decimal TB/s")
    gemm.add_argument("--l2-tbps", type=float, help="measured L2 rate in decimal TB/s")
    for name in ("m", "n", "k", "tile-m", "tile-n", "x-item-bytes", "w-item-bytes", "y-item-bytes"):
        gemm.add_argument(f"--{name}", type=int, required=True)
    gemm.add_argument("--extra-hbm-bytes", type=int, default=0)
    gemm.add_argument("--extra-l2-bytes", type=int, default=0)
    stream.add_argument("--read-bytes", type=int, required=True)
    stream.add_argument("--write-bytes", type=int, required=True)
    values = vars(parser.parse_args())
    arch = values.pop("arch")
    operator = values.pop("operator")
    default_hbm, default_l2 = REFERENCE_TBPS.get(arch, (None, None))
    for name, default in (("hbm_tbps", default_hbm), ("l2_tbps", default_l2)):
        if name in values and values[name] is None:
            if default is None:
                parser.error(f"{arch}: provide --{name.replace('_', '-')} from a fixed-clock measurement")
            values[name] = default
    try:
        result = gemm_floor(**values) if operator == "gemm" else stream_floor(**values)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"arch": arch, **result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
