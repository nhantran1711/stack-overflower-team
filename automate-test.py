#!/usr/bin/env python
import argparse
import itertools
import json

import numpy as np
import pandas as pd


PRICES_FILE = "./prices.txt"
NUM_TEST_DAYS = 250
SCORE_PARAM = 1.0

DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DLR_POS_LIMIT = 10_000
INST0_DLR_POS_LIMIT = 100_000

# Edit this grid to widen or narrow the search.
PARAM_GRID = {
    "lookbacks": [
        [3, 5, 8, 13, 20]
    ],
    "winsor_pct": [8, 9],
    "vol_target_lookback": [5],
    "vol_target_ref": [60],
    "dollar_scale": [23000, 23500, 24000],
    "inst0_multiplier": [8.0, 9],
    "turnover_deadband": [900, 1100, 1300, 1500, 1700],
}


def load_prices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def score(mu, sigma, param=SCORE_PARAM):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def charge_fees(dvolumes, comm_rate):
    return np.sum(dvolumes * comm_rate)


def make_strategy(params, ninst):
    lookbacks = params["lookbacks"]
    weights = 1.0 / np.array(lookbacks)
    max_lookback = max(lookbacks)
    prev_pos = None

    pos_multiplier = np.ones(ninst)
    pos_multiplier[0] = params["inst0_multiplier"]

    def get_position(prc_so_far):
        nonlocal prev_pos
        nins, nt = prc_so_far.shape
        min_history = max(max_lookback, params["vol_target_ref"]) + 2

        if nt < min_history:
            prev_pos = np.zeros(nins, dtype=int)
            return prev_pos

        logp = np.log(prc_so_far)
        all_rets = np.diff(logp, axis=1)

        zscores = []
        for lb in lookbacks:
            rets = all_rets[:, -lb:]
            market_ret = np.median(rets, axis=0)
            resid_rets = rets - market_ret
            cumulative_resid = resid_rets.sum(axis=1)

            lo, hi = np.percentile(
                cumulative_resid,
                [params["winsor_pct"], 100 - params["winsor_pct"]],
            )
            cumulative_resid = np.clip(cumulative_resid, lo, hi)
            zscores.append(
                (cumulative_resid - cumulative_resid.mean())
                / (cumulative_resid.std() + 1e-9)
            )

        signal = -np.average(zscores, axis=0, weights=weights)

        vol = np.maximum(all_rets[:, -max_lookback:].std(axis=1), 0.005)
        risk_adj = signal / vol
        risk_adj -= risk_adj.mean()
        risk_adj /= np.mean(np.abs(risk_adj)) + 1e-9

        recent_vol = all_rets[:, -params["vol_target_lookback"] :].std()
        ref_vol = all_rets[:, -params["vol_target_ref"] :].std()
        vol_target_adj = ref_vol / (recent_vol + 1e-9)

        target_dollars = (
            risk_adj
            * params["dollar_scale"]
            * vol_target_adj
            * pos_multiplier[:nins]
        )
        current_pos = (target_dollars / prc_so_far[:, -1]).astype(int)

        # Avoid tiny rebalances that mostly add fees.
        if prev_pos is not None and params["turnover_deadband"] > 0:
            trade_dollars = np.abs((current_pos - prev_pos) * prc_so_far[:, -1])
            keep_old = trade_dollars < params["turnover_deadband"]
            current_pos[keep_old] = prev_pos[keep_old]

        prev_pos = current_pos
        return current_pos

    return get_position


def evaluate(prices, params, start_day, end_day):
    ninst = prices.shape[0]
    get_position = make_strategy(params, ninst)

    comm_rate = np.full(ninst, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE

    dlr_pos_limit = np.full(ninst, DEFAULT_DLR_POS_LIMIT)
    dlr_pos_limit[0] = INST0_DLR_POS_LIMIT

    cash = 0.0
    cur_pos = np.zeros(ninst)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0
    daily_pl = []

    for t in range(start_day, end_day + 1):
        prc_so_far = prices[:, :t]
        cur_prices = prc_so_far[:, -1]

        if t < end_day:
            new_pos_orig = get_position(prc_so_far)
            pos_limits = (dlr_pos_limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta_pos = new_pos - cur_pos
        cash -= cur_prices.dot(delta_pos) + comm

        dvolumes = cur_prices * np.abs(delta_pos)
        tot_dvolume += np.sum(dvolumes)
        comm = charge_fees(dvolumes, comm_rate)

        cur_pos = np.array(new_pos)
        today_pl = cash + cur_pos.dot(cur_prices) - value
        value = cash + cur_pos.dot(cur_prices)

        if t > start_day:
            daily_pl.append(today_pl)

    pll = np.array(daily_pl)
    mean_pl = np.mean(pll)
    std_pl = np.std(pll)
    ann_sharpe = np.sqrt(250) * mean_pl / std_pl if std_pl > 0 else 0.0
    ret = value / tot_dvolume if tot_dvolume > 0 else 0.0

    return {
        "score": score(mean_pl, std_pl),
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": ann_sharpe,
        "return": ret,
        "dvolume": tot_dvolume,
    }


def param_combinations():
    keys = list(PARAM_GRID)
    for values in itertools.product(*(PARAM_GRID[key] for key in keys)):
        yield dict(zip(keys, values))


def make_windows(nt, num_test_days, rolling):
    if not rolling:
        return [(nt - num_test_days, nt)]

    starts = [250, 375, nt - num_test_days]
    windows = []
    for start in starts:
        end = start + num_test_days
        if 1 <= start < end <= nt:
            windows.append((start, end))

    return sorted(set(windows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=PRICES_FILE)
    parser.add_argument("--days", type=int, default=NUM_TEST_DAYS)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--rolling", action="store_true")
    args = parser.parse_args()

    prices = load_prices(args.prices)
    windows = make_windows(prices.shape[1], args.days, args.rolling)
    results = []

    for idx, params in enumerate(param_combinations(), start=1):
        window_results = [
            evaluate(prices, params, start_day, end_day)
            for start_day, end_day in windows
        ]
        avg_score = np.mean([result["score"] for result in window_results])
        last_result = window_results[-1]

        results.append(
            {
                "avg_score": avg_score,
                "last_score": last_result["score"],
                "last_mean_pl": last_result["mean_pl"],
                "last_sharpe": last_result["sharpe"],
                "last_dvolume": last_result["dvolume"],
                "params": params,
            }
        )

        if idx % 100 == 0:
            print(f"tested {idx} configs")

    results.sort(key=lambda row: row["avg_score"], reverse=True)

    print("rank,avg_score,last_score,last_mean_pl,last_sharpe,last_dvolume,params")
    for rank, row in enumerate(results[: args.top], start=1):
        print(
            f"{rank},"
            f"{row['avg_score']:.2f},"
            f"{row['last_score']:.2f},"
            f"{row['last_mean_pl']:.2f},"
            f"{row['last_sharpe']:.2f},"
            f"{row['last_dvolume']:.0f},"
            f"{json.dumps(row['params'])}"
        )


if __name__ == "__main__":
    main()
