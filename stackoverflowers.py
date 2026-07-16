import numpy as np

nInst = 51
LOOKBACKS = [3, 5, 8, 13, 20]  # ensemble
LOOKBACK_WEIGHTS = 1.0 / np.array(LOOKBACKS)  # favor shorter lookbacks
WINSOR_PCT = 8  # clip extreme

VOL_TARGET_LOOKBACK = 5  # fast read
VOL_TARGET_REF = 60  # reference vol level to target

DEADBAND = 0.2  # ignore target changes smaller than this fraction of the target

posLimitMultiplier = np.ones(nInst)
posLimitMultiplier[0] = 7.5

_state = {"pos": None, "lastNt": None}


def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    maxLookback = max(LOOKBACKS)

    if nt < max(maxLookback, VOL_TARGET_REF) + 2:
        _state["pos"] = None
        _state["lastNt"] = nt
        return np.zeros(nins, dtype=int)

    if _state["lastNt"] is not None and nt <= _state["lastNt"]:
        _state["pos"] = None  # nt didn't grow -> a new backtest run started

    logp = np.log(prcSoFar)
    allRets = np.diff(logp, axis=1)  # full return history

    zScores = []
    for lb in LOOKBACKS:
        rets = allRets[:, -lb:]
        marketRet = np.median(rets, axis=0)  # daily cross-sectional return
        residRets = rets - marketRet  # residual returns after removing cross-sectional mean

        cumulativeResid = residRets.sum(axis=1)  # cumulative residual

        # winsorise
        lo, hi = np.percentile(cumulativeResid, [WINSOR_PCT, 100 - WINSOR_PCT])
        cumulativeResid = np.clip(cumulativeResid, lo, hi)

        z = (cumulativeResid - cumulativeResid.mean()) / (cumulativeResid.std() + 1e-9)
        zScores.append(z)

    signal = -np.average(zScores, axis=0, weights=LOOKBACK_WEIGHTS)

    # vol floor
    vol = np.maximum(
        allRets[:, -maxLookback:].std(axis=1),
        0.005
    )
    riskAdj = signal / vol

    riskAdj -= np.mean(riskAdj)

    # cap extreme conviction so a few outlier scores don't hog dollar budget
    # that would just get clipped by the per-instrument position limit anyway
    riskAdj = np.clip(riskAdj, -1.5 * riskAdj.std(), 1.5 * riskAdj.std())

    # normalize riskAdj to have mean absolute value of 1
    riskAdj /= (
        np.mean(np.abs(riskAdj)) + 1e-9
    )

    # scale exposure inversely to the current vol regime vs. its recent normal level
    recentVol = allRets[:, -VOL_TARGET_LOOKBACK:].std()
    refVol = allRets[:, -VOL_TARGET_REF:].std()
    volTargetAdj = refVol / (recentVol + 1e-9)

    targetDollars = riskAdj * 22500 * volTargetAdj * posLimitMultiplier

    # convert targetDollars to integer number of shares
    lastPrice = prcSoFar[:, -1]
    targetPos = (targetDollars / lastPrice).astype(int)

    prevPos = _state["pos"]
    if prevPos is None:
        newPos = targetPos
    else:
        delta = targetPos - prevPos
        moveThreshold = np.maximum(np.abs(targetPos) * DEADBAND, 1)
        newPos = np.where(np.abs(delta) >= moveThreshold, targetPos, prevPos).astype(int)

    _state["pos"] = newPos
    _state["lastNt"] = nt
    return newPos
