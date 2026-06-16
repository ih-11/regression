#!/usr/bin/env python3

"""
Precompute comprehensive RNA secondary-structure features.

Input:
    RNA sequence table with columns:
        var_id, trans_id, gene_id, 5'UTR, CDS, 3'UTR

Output:
    TSV/TSV.GZ with IDs + structure features.

Feature naming:
    <region>.<feature>

Regions:
    5'UTR
    CDS
    3'UTR
    mRNA
    AUGwin
    Stopwin
"""

import sys
import argparse
import logging
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import RNA
import sylib


LOGGER = logging.getLogger(__name__)
logger = LOGGER


# ============================================
# Basic helpers
# ============================================

def normalize_rna(seq):
    if pd.isna(seq):
        return ""
    return str(seq).upper().replace("T", "U").replace("N", "")


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        x = float(x)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def aug_window(utr5, cds, left=30, right=30):
    utr5 = normalize_rna(utr5)
    cds = normalize_rna(cds)
    return utr5[-left:] + cds[:right]


def stop_window(cds, utr3, left=30, right=30):
    cds = normalize_rna(cds)
    utr3 = normalize_rna(utr3)
    return cds[-left:] + utr3[:right]


# ============================================
# Dot-bracket parsing
# ============================================

def parse_pairs(struct):
    stack = []
    pairs = []

    for i, ch in enumerate(struct):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            pairs.append((j, i))

    pairs.sort()
    return pairs


def depth_array(struct):
    depth = np.zeros(len(struct), dtype=float)
    d = 0

    for i, ch in enumerate(struct):
        if ch == "(":
            d += 1
            depth[i] = d
        elif ch == ")":
            depth[i] = d
            d = max(0, d - 1)
        else:
            depth[i] = d

    return depth


def stem_lengths_from_pairs(pairs):
    pair_set = set(pairs)
    stems = []

    for i, j in pairs:
        if (i - 1, j + 1) in pair_set:
            continue

        length = 1
        a, b = i + 1, j - 1

        while (a, b) in pair_set:
            length += 1
            a += 1
            b -= 1

        stems.append(length)

    return stems


def immediate_children(pair, pairs):
    i, j = pair
    inside = [
        p for p in pairs
        if i < p[0] and p[1] < j
    ]

    children = []

    for p in inside:
        pi, pj = p
        has_parent_inside = False

        for q in inside:
            qi, qj = q
            if qi < pi and pj < qj:
                has_parent_inside = True
                break

        if not has_parent_inside:
            children.append(p)

    children.sort()
    return children


def loop_features(struct, pairs):
    L = len(struct)

    hp_sizes = []
    int_sizes = []
    bulge_sizes = []
    multi_sizes = []

    for pair in pairs:
        i, j = pair
        children = immediate_children(pair, pairs)

        if len(children) == 0:
            hp_sizes.append(max(0, j - i - 1))

        elif len(children) == 1:
            a, b = children[0]
            left = max(0, a - i - 1)
            right = max(0, j - b - 1)
            size = left + right

            if left == 0 or right == 0:
                bulge_sizes.append(size)
            else:
                int_sizes.append(size)

        else:
            unpaired = 0
            prev = i + 1

            for a, b in children:
                unpaired += max(0, a - prev)
                prev = b + 1

            unpaired += max(0, j - prev)
            multi_sizes.append(unpaired)

    paired_pos = set()
    for i, j in pairs:
        paired_pos.add(i)
        paired_pos.add(j)

    external_size = sum(
        1 for i, ch in enumerate(struct)
        if ch == "." and i not in paired_pos
    )

    return {
        "hpN": len(hp_sizes),
        "hpMean": float(np.mean(hp_sizes)) if hp_sizes else 0.0,
        "hpMed": float(np.median(hp_sizes)) if hp_sizes else 0.0,
        "hpMax": float(np.max(hp_sizes)) if hp_sizes else 0.0,
        "hpMin": float(np.min(hp_sizes)) if hp_sizes else 0.0,

        "intN": len(int_sizes),
        "intMean": float(np.mean(int_sizes)) if int_sizes else 0.0,
        "intMax": float(np.max(int_sizes)) if int_sizes else 0.0,
        "intNT": float(np.sum(int_sizes)) if int_sizes else 0.0,

        "bulgeN": len(bulge_sizes),
        "bulgeMean": float(np.mean(bulge_sizes)) if bulge_sizes else 0.0,
        "bulgeMax": float(np.max(bulge_sizes)) if bulge_sizes else 0.0,
        "bulgeNT": float(np.sum(bulge_sizes)) if bulge_sizes else 0.0,

        "multiN": len(multi_sizes),
        "multiMean": float(np.mean(multi_sizes)) if multi_sizes else 0.0,
        "multiMax": float(np.max(multi_sizes)) if multi_sizes else 0.0,
        "multiNT": float(np.sum(multi_sizes)) if multi_sizes else 0.0,

        "extSize": external_size,
        "extFrac": safe_div(external_size, L),
    }


def structural_complexity_features(struct, pairs, stems):
    L = len(struct)
    d = depth_array(struct)

    n_stem = len(stems)
    stem_nt = 2 * sum(stems)
    loop_nt = max(0, L - stem_nt)

    chars = list(struct)
    probs = []

    for ch in [".", "(", ")"]:
        p = chars.count(ch) / L if L > 0 else 0.0
        if p > 0:
            probs.append(p)

    shannon = -sum(p * np.log2(p) for p in probs) if probs else 0.0

    return {
        "nestMax": float(np.max(d)) if L > 0 else 0.0,
        "nestMean": float(np.mean(d)) if L > 0 else 0.0,
        "stemLoop": safe_div(stem_nt, loop_nt),
        "loopStem": safe_div(loop_nt, stem_nt),
        "branchN": float(n_stem),
        "branchFac": safe_div(len(pairs), max(1, n_stem)),
        "cxIdx": safe_div(len(pairs) + n_stem + np.max(d) if L > 0 else 0.0, L),
        "strEnt": shannon,
    }


def pair_distance_features(pairs, L, long_frac=0.25):
    if not pairs:
        return {
            "bpDistMean": 0.0,
            "bpDistMed": 0.0,
            "bpDistMax": 0.0,
            "bpDistVar": 0.0,
            "longBpN": 0.0,
            "longBpFrac": 0.0,
        }

    dist = np.array([j - i for i, j in pairs], dtype=float)
    long_cut = max(1, L * long_frac)
    long_n = float(np.sum(dist >= long_cut))

    return {
        "bpDistMean": float(np.mean(dist)),
        "bpDistMed": float(np.median(dist)),
        "bpDistMax": float(np.max(dist)),
        "bpDistVar": float(np.var(dist)),
        "longBpN": long_n,
        "longBpFrac": safe_div(long_n, len(dist)),
    }


# ============================================
# BPP helpers
# ============================================

def extract_bpp_pairs(fc, L, cutoff=1e-6):
    pairs = []

    try:
        if hasattr(fc, "plist_from_probs"):
            plist = fc.plist_from_probs(cutoff)

            for p in plist:
                i = int(p.i)
                j = int(p.j)
                prob = float(p.p)

                if i > 0 and j > 0 and prob >= cutoff:
                    pairs.append((i - 1, j - 1, prob))

            return pairs
    except Exception:
        pass

    try:
        bpp = fc.bpp()
        arr = np.asarray(bpp, dtype=float)

        if arr.ndim == 2:
            for i in range(min(L, arr.shape[0])):
                for j in range(i + 1, min(L, arr.shape[1])):
                    prob = float(arr[i, j])
                    if prob >= cutoff:
                        pairs.append((i, j, prob))

            return pairs
    except Exception:
        pass

    return pairs


def bpp_features(bpp_pairs, L):
    if not bpp_pairs or L == 0:
        return {
            "ppMean": 0.0,
            "ppMed": 0.0,
            "ppMax": 0.0,
            "ppVar": 0.0,
            "ppN01": 0.0,
            "ppN05": 0.0,
            "ppN09": 0.0,
            "ppEnt": 0.0,
            "ppSparse": 1.0,
            "posEntMean": 0.0,
            "posEntMed": 0.0,
            "posEntMax": 0.0,
            "posEntStd": 0.0,
        }

    probs = np.array([p for _, _, p in bpp_pairs], dtype=float)

    entropy = -np.sum(
        probs * np.log2(np.clip(probs, 1e-12, 1.0))
    )

    per_pos = np.zeros(L, dtype=float)

    for i, j, p in bpp_pairs:
        per_pos[i] += p
        per_pos[j] += p

    per_pos = np.clip(per_pos, 0.0, 1.0)

    pos_entropy = -(
        per_pos * np.log2(np.clip(per_pos, 1e-12, 1.0))
        + (1 - per_pos) * np.log2(np.clip(1 - per_pos, 1e-12, 1.0))
    )

    possible_pairs = L * (L - 1) / 2

    return {
        "ppMean": float(np.mean(probs)),
        "ppMed": float(np.median(probs)),
        "ppMax": float(np.max(probs)),
        "ppVar": float(np.var(probs)),
        "ppN01": float(np.sum(probs > 0.1)),
        "ppN05": float(np.sum(probs > 0.5)),
        "ppN09": float(np.sum(probs > 0.9)),
        "ppEnt": float(entropy),
        "ppSparse": 1.0 - safe_div(len(probs), possible_pairs),
        "posEntMean": float(np.mean(pos_entropy)),
        "posEntMed": float(np.median(pos_entropy)),
        "posEntMax": float(np.max(pos_entropy)),
        "posEntStd": float(np.std(pos_entropy)),
    }


# ============================================
# Region feature calculation
# ============================================

def eval_region_structure(seq, region, temperature=None):
    seq = normalize_rna(seq)
    L = len(seq)

    prefix = f"{region}."

    out = {}

    if L == 0:
        zero_features = [
            "Len", "MFE", "MFEnt", "EFE", "CentE", "MEAE",
            "dMFE_EFE", "dMFE_Cent", "pMFE", "EDiv",
            "pairN", "unpairN", "fracP", "fracU",
            "ppMean", "ppMed", "ppMax", "ppVar",
            "ppN01", "ppN05", "ppN09", "ppEnt", "ppSparse",
            "posEntMean", "posEntMed", "posEntMax", "posEntStd",
            "stemN", "stemMean", "stemMed", "stemMax", "stemMin",
            "stemVar", "stemNT", "stemFrac",
            "hpN", "hpMean", "hpMed", "hpMax", "hpMin",
            "intN", "intMean", "intMax", "intNT",
            "bulgeN", "bulgeMean", "bulgeMax", "bulgeNT",
            "multiN", "multiMean", "multiMax", "multiNT",
            "extSize", "extFrac",
            "nestMax", "nestMean", "stemLoop", "loopStem",
            "branchN", "branchFac", "cxIdx", "strEnt",
            "bpDistMean", "bpDistMed", "bpDistMax", "bpDistVar",
            "longBpN", "longBpFrac",
        ]

        return {
            prefix + name: 0.0
            for name in zero_features
        }

    md = RNA.md()
    if temperature is not None:
        md.temperature = temperature

    fc = RNA.fold_compound(seq)
    fc.params_reset(md)

    mfe_struct, mfe = fc.mfe()
    mfe = safe_float(mfe)

    efe = 0.0
    mfe_freq = 0.0
    ediv = 0.0
    centroid_e = 0.0
    mea_e = 0.0

    bpp_pairs = []

    try:
        _, efe = fc.pf()
        efe = safe_float(efe)
    except Exception:
        efe = 0.0

    try:
        mfe_freq = safe_float(fc.pr_structure(mfe_struct))
    except Exception:
        mfe_freq = np.nan

    try:
        ediv = safe_float(fc.mean_bp_distance())
    except Exception:
        ediv = 0.0

    try:
        centroid_struct, _ = fc.centroid()
        centroid_e = safe_float(fc.eval_structure(centroid_struct))
    except Exception:
        centroid_e = 0.0

    try:
        if hasattr(fc, "MEA"):
            mea_struct, _ = fc.MEA()
        else:
            mea_struct, _ = fc.mea()
        mea_e = safe_float(fc.eval_structure(mea_struct))
    except Exception:
        mea_e = 0.0

    bpp_pairs = extract_bpp_pairs(fc, L)

    pairs = parse_pairs(mfe_struct)
    stems = stem_lengths_from_pairs(pairs)

    pair_n = 2 * len(pairs)
    unpair_n = L - pair_n

    stem_nt = 2 * sum(stems)

    out[prefix + "Len"] = float(L)
    out[prefix + "MFE"] = mfe
    out[prefix + "MFEnt"] = safe_div(mfe, L)
    out[prefix + "EFE"] = efe
    out[prefix + "CentE"] = centroid_e
    out[prefix + "MEAE"] = mea_e
    out[prefix + "dMFE_EFE"] = mfe - efe
    out[prefix + "dMFE_Cent"] = mfe - centroid_e
    out[prefix + "pMFE"] = mfe_freq
    out[prefix + "EDiv"] = ediv

    out[prefix + "pairN"] = float(pair_n)
    out[prefix + "unpairN"] = float(unpair_n)
    out[prefix + "fracP"] = safe_div(pair_n, L)
    out[prefix + "fracU"] = safe_div(unpair_n, L)

    out.update({
        prefix + k: v
        for k, v in bpp_features(bpp_pairs, L).items()
    })

    out[prefix + "stemN"] = float(len(stems))
    out[prefix + "stemMean"] = float(np.mean(stems)) if stems else 0.0
    out[prefix + "stemMed"] = float(np.median(stems)) if stems else 0.0
    out[prefix + "stemMax"] = float(np.max(stems)) if stems else 0.0
    out[prefix + "stemMin"] = float(np.min(stems)) if stems else 0.0
    out[prefix + "stemVar"] = float(np.var(stems)) if stems else 0.0
    out[prefix + "stemNT"] = float(stem_nt)
    out[prefix + "stemFrac"] = safe_div(stem_nt, L)

    out.update({
        prefix + k: v
        for k, v in loop_features(mfe_struct, pairs).items()
    })

    out.update({
        prefix + k: v
        for k, v in structural_complexity_features(mfe_struct, pairs, stems).items()
    })

    out.update({
        prefix + k: v
        for k, v in pair_distance_features(pairs, L).items()
    })

    return out


# ============================================
# Cross-region features
# ============================================

def region_of_pos(pos, len5, len_cds, len3):
    if pos < len5:
        return "5"
    elif pos < len5 + len_cds:
        return "C"
    else:
        return "3"


def cross_region_features(seq5, cds, seq3, temperature=None):
    seq5 = normalize_rna(seq5)
    cds = normalize_rna(cds)
    seq3 = normalize_rna(seq3)

    mrna = seq5 + cds + seq3
    L = len(mrna)

    prefix = "mRNA."

    out = {
        prefix + "x5C_N": 0.0,
        prefix + "xC3_N": 0.0,
        prefix + "x53_N": 0.0,
        prefix + "xFrac": 0.0,
        prefix + "xPPMean": 0.0,
        prefix + "xPPMax": 0.0,
    }

    if L == 0:
        return out

    md = RNA.md()
    if temperature is not None:
        md.temperature = temperature

    fc = RNA.fold_compound(mrna)
    fc.params_reset(md)

    try:
        mfe_struct, _ = fc.mfe()
        pairs = parse_pairs(mfe_struct)
    except Exception:
        pairs = []

    n5 = nC = n3 = 0
    cross_total = 0

    for i, j in pairs:
        ri = region_of_pos(i, len(seq5), len(cds), len(seq3))
        rj = region_of_pos(j, len(seq5), len(cds), len(seq3))

        if ri == rj:
            continue

        cross_total += 1

        pair_type = "".join(sorted([ri, rj]))

        if pair_type == "5C":
            n5 += 1
        elif pair_type == "3C":
            nC += 1
        elif pair_type == "35":
            n3 += 1

    out[prefix + "x5C_N"] = float(n5)
    out[prefix + "xC3_N"] = float(nC)
    out[prefix + "x53_N"] = float(n3)
    out[prefix + "xFrac"] = safe_div(cross_total, len(pairs))

    try:
        fc.pf()
        bpp_pairs = extract_bpp_pairs(fc, L)
        cross_probs = []

        for i, j, p in bpp_pairs:
            ri = region_of_pos(i, len(seq5), len(cds), len(seq3))
            rj = region_of_pos(j, len(seq5), len(cds), len(seq3))

            if ri != rj:
                cross_probs.append(p)

        if cross_probs:
            out[prefix + "xPPMean"] = float(np.mean(cross_probs))
            out[prefix + "xPPMax"] = float(np.max(cross_probs))

    except Exception:
        pass

    return out


# ============================================
# Row-level worker
# ============================================

def compute_row_features(row, temperature=None):
    seq5 = normalize_rna(row.get("5'UTR", ""))
    cds = normalize_rna(row.get("CDS", ""))
    seq3 = normalize_rna(row.get("3'UTR", ""))

    mrna = seq5 + cds + seq3

    regions = {
        "5'UTR": seq5,
        "CDS": cds,
        "3'UTR": seq3,
        "mRNA": mrna,
        "AUGwin": aug_window(seq5, cds),
        "Stopwin": stop_window(cds, seq3),
    }

    out = {}

    for region, seq in regions.items():
        out.update(
            eval_region_structure(
                seq,
                region,
                temperature=temperature,
            )
        )

    out.update(
        cross_region_features(
            seq5,
            cds,
            seq3,
            temperature=temperature,
        )
    )

    return out


# ============================================
# Main
# ============================================

def main(
    input_file,
    output_file,
    temperature=None,
    n_processors=1,
    force=False,
):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if output_file.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_file}"
        )

    logger.info("Loading input: %s", input_file)

    data_df, metadata = sylib.fileio.load_df(str(input_file))
    metadata.print_minimum_data(
        label="Input data",
        logger=logger,
        logging_level="info",
    )

    id_cols = [
        col for col in ["var_id", "trans_id", "gene_id"]
        if col in data_df.columns
    ]

    rows = data_df.to_dict("records")

    logger.info(
        "Calculating structure features: n=%s, processors=%s",
        len(rows),
        n_processors,
    )

    if n_processors == 1:
        feature_rows = []

        for i, row in enumerate(rows, start=1):
            feature_rows.append(
                compute_row_features(row, temperature=temperature)
            )

            if i % 100 == 0 or i == len(rows):
                logger.info("completed %s / %s", i, len(rows))

    else:
        feature_rows = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processors) as executor:
            futures = executor.map(
                compute_row_features,
                rows,
                [temperature] * len(rows),
            )

            for i, result in enumerate(futures, start=1):
                feature_rows.append(result)

                if i % 100 == 0 or i == len(rows):
                    logger.info("completed %s / %s", i, len(rows))

    feature_df = pd.DataFrame(feature_rows)

    out_df = pd.concat(
        [
            data_df[id_cols].reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )

    logger.info("Writing output: %s", output_file)

    metadata = sylib.fileio.write_df(
        out_df,
        str(output_file),
        index=False,
    )

    metadata.print_minimum_data(
        label="Output data",
        logger=logger,
        logging_level="info",
    )

    logger.info("Done.")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Precompute comprehensive RNA secondary-structure features."
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Input RNA sequence table.",
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Output structure feature table.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="RNA folding temperature. Example: 22 or 25.",
    )

    parser.add_argument(
        "-p",
        "--n-processors",
        type=int,
        default=1,
        help="Number of parallel processes.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists.",
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)8s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        temperature=args.temperature,
        n_processors=args.n_processors,
        force=args.force,
    )