#!/usr/bin/env python3

import argparse
import concurrent.futures
import logging
from pathlib import Path

import pandas as pd
import RNA
import sylib


logger = logging.getLogger(__name__)


def normalize_rna(seq):
    if pd.isna(seq):
        return ""
    return str(seq).upper().replace("T", "U")


def calc_mfe(seq, temperature):
    seq = normalize_rna(seq)

    if len(seq) == 0:
        return 0.0

    md = RNA.md(temperature=temperature)
    fc = RNA.fold_compound(seq)
    fc.params_reset(md)
    _, mfe = fc.mfe()

    return float(mfe)


def parallel_calc_mfe(seq_list, temperature, n_workers):
    mfe_list = [None] * len(seq_list)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(calc_mfe, seq, temperature): i
            for i, seq in enumerate(seq_list)
        }

        for n_done, future in enumerate(
            concurrent.futures.as_completed(future_to_idx),
            start=1,
        ):
            i = future_to_idx[future]
            mfe_list[i] = future.result()

            if n_done % 100 == 0 or n_done == len(seq_list):
                logger.info("completed %s / %s", n_done, len(seq_list))

    return mfe_list


def main(input_file, output_file, temperature, n_workers, force=False):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if output_file.exists() and not force:
        raise FileExistsError(f"Output file already exists: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading input: %s", input_file)

    df, metadata = sylib.fileio.load_df(str(input_file))
    metadata.print_minimum_data(
        label="Input data",
        logger=logger,
        logging_level="info",
    )

    required_cols = ["var_id", "trans_id", "gene_id", "5'UTR", "CDS", "3'UTR"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    mrna_list = [
        normalize_rna(u5) + normalize_rna(cds) + normalize_rna(u3)
        for u5, cds, u3 in zip(df["5'UTR"], df["CDS"], df["3'UTR"])
    ]

    out_df = df[["var_id", "trans_id", "gene_id"]].copy()

    logger.info("calculating 5'UTR.MFE")
    out_df["5'UTR.MFE"] = parallel_calc_mfe(
        df["5'UTR"].tolist(),
        temperature,
        n_workers,
    )

    logger.info("calculating CDS.MFE")
    out_df["CDS.MFE"] = parallel_calc_mfe(
        df["CDS"].tolist(),
        temperature,
        n_workers,
    )

    logger.info("calculating 3'UTR.MFE")
    out_df["3'UTR.MFE"] = parallel_calc_mfe(
        df["3'UTR"].tolist(),
        temperature,
        n_workers,
    )

    logger.info("calculating mRNA.MFE")
    out_df["mRNA.MFE"] = parallel_calc_mfe(
        mrna_list,
        temperature,
        n_workers,
    )

    logger.info("writing output: %s", output_file)
    sylib.fileio.write_df(out_df, str(output_file), index=False)

    logger.info("done")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute MFE features for regression benchmark."
    )

    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("-p", "--n-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)8s: %(message)s",
    )

    args = parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
        temperature=args.temperature,
        n_workers=args.n_workers,
        force=args.force,
    )