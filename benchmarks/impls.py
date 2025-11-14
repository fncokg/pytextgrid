from functools import partial

import parselmouth
import textgrid as textgrid_pkg
import gridio as gridio_pkg

import pandas as pd


def gridio_construct(fnames):
    return gridio_pkg.TextGrid.from_file(fnames)


def textgrid_construct(fnames):
    return list(map(textgrid_pkg.TextGrid.fromFile, fnames))


def parselmouth_construct(fnames):
    data_list = []
    for fname in fnames:
        tgt = parselmouth.TextGrid.read(str(fname))
        Praat = partial(parselmouth.praat.call, tgt)

        tgt_data = {
            "xmin": Praat("Get start time"),
            "xmax": Praat("Get end time"),
            "tiers": [],
        }
        ntier = Praat("Get number of tiers")
        for tier_idx in range(1, ntier + 1):
            tier = {
                "name": Praat("Get tier name", tier_idx),
                "is_interval": Praat("Is interval tier", tier_idx),
                "items": [],
            }
            if tier["is_interval"]:
                nitems = Praat("Get number of intervals", tier_idx)
                for j in range(1, nitems + 1):
                    tier["items"].append(
                        {
                            "tmin": Praat("Get start time of interval", tier_idx, j),
                            "tmax": Praat("Get end time of interval", tier_idx, j),
                            "label": Praat("Get label of interval", tier_idx, j),
                        }
                    )
            else:
                nitems = Praat("Get number of points", tier_idx)
                for j in range(1, nitems + 1):
                    tier["items"].append(
                        {
                            "point": Praat("Get time of point", tier_idx, j),
                            "label": Praat("Get label of point", tier_idx, j),
                        }
                    )
            tgt_data["tiers"].append(tier)
        data_list.append(tgt_data)
    return data_list


def gridio_to_df(fnames):
    return gridio_pkg.textgrid_to_df(fnames, strict=False, file_type="long")


def textgrid_to_df(fnames):
    tgts = textgrid_construct(fnames)
    rows = []
    for fname, tgt in zip(fnames, tgts):
        for tier in tgt.tiers:
            is_interval = isinstance(tier, textgrid_pkg.IntervalTier)
            for item in tier:
                if is_interval:
                    tmin, tmax, label = (
                        item.minTime,
                        item.maxTime,
                        item.mark,
                    )
                else:
                    tmin, tmax, label = item.time, item.time, item.mark
                rows.append(
                    {
                        "tmin": tmin,
                        "tmax": tmax,
                        "label": label,
                        "tier": tier.name,
                        "is_interval": is_interval,
                        "filename": fname,
                    }
                )
    return pd.DataFrame(rows)


def parselmouth_to_df(fnames):
    tgts = parselmouth_construct(fnames)
    rows = []
    for fname, tgt in zip(fnames, tgts):
        for tier in tgt["tiers"]:
            is_interval = tier["is_interval"]
            for item in tier["items"]:
                if is_interval:
                    tmin, tmax, label = item["tmin"], item["tmax"], item["label"]
                else:
                    tmin, tmax, label = item["point"], item["point"], item["label"]
                rows.append(
                    {
                        "tmin": tmin,
                        "tmax": tmax,
                        "label": label,
                        "tier": tier["name"],
                        "is_interval": is_interval,
                        "filename": fname,
                    }
                )
    return pd.DataFrame(rows)
