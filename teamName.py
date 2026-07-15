import numpy as np

nInst = 51
LOOKBACKS = [3, 5, 8, 13, 20]  # ensemble
LOOKBACK_WEIGHTS = 1.0 / np.array(LOOKBACKS)  # favor shorter lookbacks
WINSOR_PCT = 8  # clip extreme 

VOL_TARGET_LOOKBACK = 5 # fast read
VOL_TARGET_REF = 60 # reference  vol level to target

# instrument 0 gets a 10x larger position limit
posLimitMultiplier = np.ones(nInst)
posLimitMultiplier[0] = 10.0

def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    maxLookback = max(LOOKBACKS)

    if nt < max(maxLookback, VOL_TARGET_REF) + 2:
        return np.zeros(nins, dtype = int)

    logp = np.log(prcSoFar)
    allRets = np.diff(logp, axis = 1) # full return history

    zScores = []
    for lb in LOOKBACKS:
        rets = allRets[:, -lb:]
        marketRet = np.median(rets, axis = 0) # daily cross-sectional return
        residRets = rets - marketRet # residual returns after removing cross-sectional mean

        cumulativeResid = residRets.sum(axis = 1) # cumulative residual

        # winsorise
        lo, hi = np.percentile(cumulativeResid, [WINSOR_PCT, 100 - WINSOR_PCT])
        cumulativeResid = np.clip(cumulativeResid, lo, hi)

        z = (cumulativeResid - cumulativeResid.mean()) / (cumulativeResid.std() + 1e-9)
        zScores.append(z)

    signal = -np.average(zScores, axis = 0, weights = LOOKBACK_WEIGHTS)

    # vol floor 
    vol = np.maximum(
        allRets[:, -maxLookback:].std(axis = 1),
        0.005
    )
    riskAdj = signal / vol

    # normalize riskAdj to have mean absolute value of 1
    riskAdj -= np.mean(riskAdj)

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
    currentPos = (targetDollars / lastPrice).astype(int)
    return currentPos
