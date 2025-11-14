from pathlib import Path
from gridio import data_to_textgrid


def generate_items(
    count: int, step: float, offset: float, prefix: str, is_interval: bool
):
    items = []
    for idx in range(count):
        start = offset + idx * step
        end = start + step if is_interval else start
        items.append((start, end, f"{prefix}_{idx}"))
    return items


def build_textgrid_data(
    n_interval_tiers: int,
    n_point_tiers: int,
    interval_items: int,
    point_items: int,
    step: float = 0.01,
):
    tiers = []
    tmax = (
        (max(interval_items, point_items) * step)
        if max(interval_items, point_items)
        else 0.0
    )

    for tid in range(n_interval_tiers):
        tiers.append(
            (
                f"interval_{tid}",
                True,
                generate_items(interval_items, step, 0.0, f"i{tid}", True),
            )
        )
    for tid in range(n_point_tiers):
        tiers.append(
            (
                f"point_{tid}",
                False,
                generate_items(point_items, step, step / 2.0, f"p{tid}", False),
            )
        )
    return (0.0, tmax, tiers)


def build_files(
    output_dir: Path,
    file_count: int,
    interval_tiers: int,
    point_tiers: int,
    interval_items: int,
    point_items: int,
    step: float = 0.01,
):
    payload = build_textgrid_data(
        interval_tiers, point_tiers, interval_items, point_items, step
    )
    files = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(file_count):
        file_path = output_dir / f"sample_{idx:04d}.TextGrid"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_to_textgrid(payload, str(file_path), file_type="long")
        files.append(file_path)

    return files
