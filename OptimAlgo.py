"""
Erika Antoinette Tan's Optimization Algorithm implemented for Datstat Consulting
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional, Dict

# SciPy is used for robust box-constrained local search (exploitation).
# If not available, we fall back to a budgeted random local search.
try:
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    _scipy_minimize = None  # type: ignore


# ------------------------------- Utilities ------------------------------- #

def LatinHypercube(nSamples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube sample in [0,1]^dim."""
    cuts = np.linspace(0.0, 1.0, nSamples + 1)
    u = rng.random((nSamples, dim))
    a = cuts[:-1]
    b = cuts[1:]
    pts = a[:, None] + u * (b - a)[:, None]
    for j in range(dim):
        rng.shuffle(pts[:, j])
    return pts


def LhsInBounds(nSamples: int, xMin: np.ndarray, xMax: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube sample mapped to [xMin, xMax]."""
    dim = xMin.size
    base = LatinHypercube(nSamples, dim, rng)
    return xMin + base * (xMax - xMin)


def ApplyBounds(x: np.ndarray, xMin: np.ndarray, xMax: np.ndarray) -> np.ndarray:
    """Clamp x to [xMin, xMax]."""
    return np.minimum(xMax, np.maximum(xMin, x))


def HandleBoundary(x: np.ndarray, xMin: np.ndarray, xMax: np.ndarray, xParent: np.ndarray,
                   strategy: int, rng: np.random.Generator) -> np.ndarray:
    """
    Boundary handling strategies:
      1) Midpoint to bound (DE-style)
      2) Reflection (then clamp)
      3) Random reinit for violated components
    """
    x = x.copy()
    if strategy == 1:
        pos = x < xMin
        x[pos] = 0.5 * (xParent[pos] + xMin[pos])
        pos = x > xMax
        x[pos] = 0.5 * (xParent[pos] + xMax[pos])
    elif strategy == 2:
        pos = x < xMin
        x[pos] = np.minimum(xMax[pos], np.maximum(xMin[pos], 2 * xMin[pos] - xParent[pos]))
        pos = x > xMax
        x[pos] = np.maximum(xMin[pos], np.minimum(xMax[pos], 2 * xMax[pos] - xParent[pos]))
    else:
        pos = x < xMin
        if np.any(pos):
            x[pos] = xMin[pos] + rng.random(np.count_nonzero(pos)) * (xMax[pos] - xMin[pos])
        pos = x > xMax
        if np.any(pos):
            x[pos] = xMin[pos] + rng.random(np.count_nonzero(pos)) * (xMax[pos] - xMin[pos])
    return x


def LevyStep(dim: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Mantegna's Lévy flight step."""
    from scipy.special import gamma as _gamma  # local import; if SciPy missing, user should install for Lévy
    sigma = (_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0.0, sigma, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    return u / (np.abs(v) ** (1.0 / beta))


# ------------------------- Function evaluation counter ------------------------- #

class FunctionCounter:
    def __init__(self, func: Callable[[np.ndarray], float]):
        self.func = func
        self.Count = 0

    def __call__(self, x: np.ndarray) -> float:
        self.Count += 1
        return float(self.func(np.asarray(x)))


# ------------------------------- Config / Result ------------------------------- #

@dataclass
class PeoaConfig:
    InitEagleSize: Optional[int] = None     # default: 20 * D^2
    LocalFoodSize: Optional[int] = None     # default: 10 * D^2 (used as local solver maxfun)
    MaxFunctionEvals: Optional[int] = None  # default: 10000 * D
    Rho: float = 0.04                       # territory radius factor
    MinPopulationSize: int = 5
    ArchiveRate: float = 1.0                # archive size = ArchiveRate * InitEagleSize
    MemorySize: int = 5                     # JADE-style Fi memory
    CauchyScale: float = 0.1                # Fi sampling scale
    LevyBeta: float = 1.5                   # Lévy exponent
    Seed: Optional[int] = None


@dataclass
class PeoaResult:
    XBest: np.ndarray
    FBest: float
    EvalsUsed: int
    History: Dict[str, List[float]] = field(default_factory=dict)


# ------------------------------- Optimizer ------------------------------- #

class PhilippineEagleOptimizer:
    def __init__(self,
                 dimension: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 objective: Callable[[np.ndarray], float],
                 config: Optional[PeoaConfig] = None):
        self.D = int(dimension)
        self.XMin = np.asarray(bounds[0], dtype=float)
        self.XMax = np.asarray(bounds[1], dtype=float)
        assert self.XMin.shape == (self.D,) and self.XMax.shape == (self.D,)
        assert np.all(self.XMin < self.XMax)

        self.ObjectiveRaw = objective
        self.Objective = FunctionCounter(objective)

        self.Config = config or PeoaConfig()
        self.Rng = np.random.default_rng(self.Config.Seed)

        # Defaults per paper
        if self.Config.InitEagleSize is None:
            self.Config.InitEagleSize = 20 * self.D * self.D
        if self.Config.LocalFoodSize is None:
            self.Config.LocalFoodSize = 10 * self.D * self.D
        if self.Config.MaxFunctionEvals is None:
            self.Config.MaxFunctionEvals = 10000 * self.D

        # Territory radius (min size 1)
        self.TerritoryRadius = max(self.Config.Rho * float(np.min(self.XMax - self.XMin)), 1.0)

        # Success-history memory for Fi
        self.H = self.Config.MemorySize
        self.MeanF = np.full(self.H, 0.2, dtype=float)
        self.MeanFIndex = 0

        # Operator probability vector
        self.P = np.array([1/3, 1/3, 1/3], dtype=float)

        # External archive
        self.Archive: List[np.ndarray] = []
        self.ArchiveMax = int(max(1, self.Config.ArchiveRate * self.Config.InitEagleSize))

        # History tracking
        self.History = {
            "BestF": [],
            "PopulationSize": [],
            "P1": [],
            "P2": [],
            "P3": [],
            "Evals": [],
        }

    # -------------------------- Local Search (Exploitation) -------------------------- #

    def LocalSearch(self, x0: np.ndarray, remainingBudget: int) -> Tuple[np.ndarray, float, int]:
        """Box-constrained local search around x0 within territory bounds."""
        if remainingBudget <= 0:
            # Consume one evaluation to keep accounting predictable
            return x0, self.Objective(x0), 1

        # Territory bounds around x0
        yMin = np.maximum(self.XMin, x0 - self.TerritoryRadius)
        yMax = np.minimum(self.XMax, x0 + self.TerritoryRadius)
        bounds = list(zip(yMin, yMax))

        startEval = self.Objective.Count
        maxFun = int(max(1, min(self.Config.LocalFoodSize, remainingBudget)))

        if _HAVE_SCIPY:
            def fun(z):
                return self.Objective(z)

            res = _scipy_minimize(fun, x0=x0, method="L-BFGS-B", bounds=bounds,
                                  options=dict(maxfun=maxFun, maxiter=maxFun, ftol=1e-15, gtol=1e-8))
            xOpt = np.asarray(res.x, dtype=float)
            fOpt = float(res.fun)
            evalsUsed = self.Objective.Count - startEval
            if evalsUsed <= 0:
                fOpt = self.Objective(xOpt)
                evalsUsed = self.Objective.Count - startEval
            return xOpt, fOpt, evalsUsed

        # Fallback: budgeted random local search inside territory
        xBest = ApplyBounds(x0.copy(), yMin, yMax)
        fBest = self.Objective(xBest)
        evalsUsed = 1
        for _ in range(maxFun - 1):
            cand = yMin + self.Rng.random(self.D) * (yMax - yMin)
            fCand = self.Objective(cand)
            evalsUsed += 1
            if fCand < fBest:
                xBest, fBest = cand, fCand
        return xBest, fBest, evalsUsed

    # -------------------------- JADE-Style Fi Adaptation -------------------------- #

    def SampleF(self, size: int) -> np.ndarray:
        Fi = np.empty(size, dtype=float)
        for i in range(size):
            j = int(self.Rng.integers(0, self.H))
            val = self.Rng.standard_cauchy() * self.Config.CauchyScale + self.MeanF[j]
            tries = 0
            while val <= 0 and tries < 8:
                val = self.Rng.standard_cauchy() * self.Config.CauchyScale + self.MeanF[j]
                tries += 1
            if val <= 0:
                val = 1e-6
            Fi[i] = min(1.0, float(val))
        return Fi

    def UpdateMeanF(self, fSuccess: List[float], improvement: List[float]) -> None:
        if not fSuccess:
            return
        FArr = np.asarray(fSuccess, dtype=float)
        w = np.asarray(improvement, dtype=float)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 0:
            return
        w /= s
        num = float(np.sum(w * FArr * FArr))
        den = float(np.sum(w * FArr))
        if den <= 0:
            return
        newMu = num / den
        self.MeanF[self.MeanFIndex] = newMu
        self.MeanFIndex = (self.MeanFIndex + 1) % self.H

    # -------------------------- Operators -------------------------- #

    def NearestNeighborIndex(self, X: np.ndarray, i: int) -> Tuple[int, float]:
        xi = X[i]
        diff = X - xi
        d2 = np.einsum("ij,ij->i", diff, diff, optimize=True)
        d2[i] = np.inf
        j = int(np.argmin(d2))
        return j, float(np.sqrt(d2[j]))

    def MovementOperator(self, X: np.ndarray, i: int, xBest: np.ndarray, Fi: float) -> np.ndarray:
        PS = X.shape[0]
        # pick r1 != i
        popIdx = [idx for idx in range(PS) if idx != i]
        r1 = int(self.Rng.choice(popIdx))
        Xr1 = X[r1]

        # pick archive candidate from pop U archive (not i, r1)
        pool: List[np.ndarray] = [X[k] for k in popIdx if k != r1]
        pool.extend(self.Archive)
        Xarc = Xr1 if len(pool) == 0 else pool[int(self.Rng.integers(0, len(pool)))]

        jNear, d = self.NearestNeighborIndex(X, i)
        Xnear = X[jNear]

        Xi = X[i]
        proximity = np.exp(-(d ** 2)) * (Xnear - Xi)
        trial = Xi + Fi * (xBest - Xi + Xr1 - Xarc + proximity)
        return trial

    def MutationIOperator(self, X: np.ndarray, i: int, xBest: np.ndarray, Fi: float) -> np.ndarray:
        PS = X.shape[0]
        idxs = [idx for idx in range(PS) if idx != i]
        if len(idxs) < 2:
            return self.XMin + self.Rng.random(self.D) * (self.XMax - self.XMin)
        r1, r2 = self.Rng.choice(idxs, size=2, replace=False)
        Xr1, Xr2 = X[r1], X[r2]
        S = self.Rng.random(self.D)
        # Lévy flight; if SciPy's gamma isn't available, this will error. Users should install SciPy or replace with heavy-tailed step.
        try:
            step = LevyStep(self.D, self.Config.LevyBeta, self.Rng)
        except Exception:
            # Fallback: heavy-tailed Cauchy step
            step = self.Rng.standard_cauchy(self.D)
        trial = Fi * (Xr1 + xBest - Xr2) + S * step
        return trial

    def MutationIIOperator(self, X: np.ndarray, i: int, xBest: np.ndarray, Fi: float) -> np.ndarray:
        xMean = X.mean(axis=0)
        xHat = self.XMin + self.Rng.random(self.D) * (self.XMax - self.XMin)
        return Fi * (xHat + xBest - xMean)

    # -------------------------- Main Run -------------------------- #

    def Run(self) -> PeoaResult:
        D = self.D
        S0 = int(self.Config.InitEagleSize)
        SMin = int(self.Config.MinPopulationSize)
        NMax = int(self.Config.MaxFunctionEvals)

        # Initialization via LHS
        X = LhsInBounds(S0, self.XMin, self.XMax, self.Rng)
        fX = np.array([self.Objective(x) for x in X], dtype=float)
        N = self.Objective.Count

        # Sort and get initial best
        order = np.argsort(fX)
        X, fX = X[order], fX[order]
        xBest = X[0].copy()
        fBest = float(fX[0])

        # First Local Phase
        remaining = NMax - N
        xLocal, fLocal, used = self.LocalSearch(xBest, remaining)
        N += used
        if fLocal < fBest:
            self.Archive.append(xBest.copy())
            X[0], fX[0] = xLocal, fLocal
            order = np.argsort(fX)
            X, fX = X[order], fX[order]
            xBest, fBest = X[0].copy(), float(fX[0])
        lastLocalBest = X[0].copy()

        # Log history
        def Log():
            self.History["BestF"].append(fBest)
            self.History["PopulationSize"].append(X.shape[0])
            self.History["P1"].append(self.P[0])
            self.History["P2"].append(self.P[1])
            self.History["P3"].append(self.P[2])
            self.History["Evals"].append(N)

        Log()

        # Generational loop
        while N < NMax:
            PS = X.shape[0]
            if PS < SMin:
                # Ensure minimum population
                need = SMin - PS
                extra = LhsInBounds(need, self.XMin, self.XMax, self.Rng)
                X = np.vstack([X, extra])
                fX = np.concatenate([fX, [self.Objective(x) for x in extra]])
                N = self.Objective.Count
                order = np.argsort(fX)
                X, fX = X[order], fX[order]

            FiAll = self.SampleF(PS)

            # Assign operators by probability vector P
            u = self.Rng.random(PS)
            cumP = np.cumsum(self.P)
            assignment = np.where(u < cumP[0], 0, np.where(u < cumP[1], 1, 2))

            # Track improvements for probability adaptation and Fi memory
            fOld = fX.copy()
            opDenom = [0.0, 0.0, 0.0]
            opGain = [0.0, 0.0, 0.0]
            fSuccess: List[float] = []
            dfSuccess: List[float] = []

            XNew = X.copy()
            fXNew = fX.copy()

            for i in range(PS):
                Fi = FiAll[i]
                op = int(assignment[i])
                xi = X[i]

                if op == 0:
                    trial = self.MovementOperator(X, i, xBest, Fi)
                elif op == 1:
                    trial = self.MutationIOperator(X, i, xBest, Fi)
                else:
                    trial = self.MutationIIOperator(X, i, xBest, Fi)

                hb = int(self.Rng.integers(1, 4))
                trial = HandleBoundary(trial, self.XMin, self.XMax, xi, hb, self.Rng)
                trial = ApplyBounds(trial, self.XMin, self.XMax)

                fTrial = self.Objective(trial)
                N = self.Objective.Count

                opDenom[op] += fOld[i]
                gain = max(0.0, fOld[i] - fTrial)
                opGain[op] += gain

                if fTrial < fOld[i]:
                    self.Archive.append(xi.copy())
                    XNew[i] = trial
                    fXNew[i] = fTrial
                    fSuccess.append(Fi)
                    dfSuccess.append(fOld[i] - fTrial)

                # Keep archive within bound
                if len(self.Archive) > self.ArchiveMax:
                    drop = len(self.Archive) - self.ArchiveMax
                    for _ in range(drop):
                        j = int(self.Rng.integers(0, len(self.Archive)))
                        self.Archive.pop(j)

                if N >= NMax:
                    break

            X, fX = XNew, fXNew

            # Update operator probabilities P
            R = np.zeros(3, dtype=float)
            for j in range(3):
                R[j] = (opGain[j] / opDenom[j]) if opDenom[j] > 0 else 0.0
            sumR = float(R.sum())
            if sumR > 0:
                Pnew = R / sumR
                Pnew = np.clip(Pnew, 0.1, 0.9)
                self.P = Pnew / Pnew.sum()

            # Update Fi memory
            self.UpdateMeanF(fSuccess, dfSuccess)

            # Sort population
            order = np.argsort(fX)
            X, fX = X[order], fX[order]

            # Linear population size reduction
            targetSize = int(round(S0 + (SMin - S0) * (N / max(1, NMax))))
            targetSize = max(SMin, min(S0, targetSize))
            if X.shape[0] > targetSize:
                X = X[:targetSize]
                fX = fX[:targetSize]

            # Update global best
            if fX[0] < fBest:
                fBest = float(fX[0])
                xBest = X[0].copy()

            # Next Local Phase
            remaining = NMax - N
            start = X[0] if not np.allclose(X[0], lastLocalBest) else lastLocalBest
            xLocal, fLocal, used = self.LocalSearch(start, remaining)
            N += used
            if fLocal < fX[0]:
                self.Archive.append(X[0].copy())
                X[0], fX[0] = xLocal, fLocal
                order = np.argsort(fX)
                X, fX = X[order], fX[order]

            if fX[0] < fBest:
                fBest = float(fX[0])
                xBest = X[0].copy()
            lastLocalBest = X[0].copy()

            # Log
            Log()
            if N >= NMax:
                break

        bestIdx = int(np.argmin(fX))
        return PeoaResult(
            XBest=X[bestIdx].copy(),
            FBest=float(fX[bestIdx]),
            EvalsUsed=int(N),
            History=self.History
        )


# ------------------------------- Convenience Entry ------------------------------- #

def PhilippineEagleAlgorithm(D: int,
                             f: Callable[[np.ndarray], float],
                             SpaceXMax: np.ndarray,
                             SpaceXMin: np.ndarray,
                             IES: Optional[int] = None,
                             IFS: Optional[int] = None,
                             MFE: Optional[int] = None,
                             Seed: Optional[int] = None) -> PeoaResult:
    """
      - D: problem dimension
      - f: objective (minimize)
      - SpaceXMax, SpaceXMin: bounds (1 x D vectors)
      - IES: InitEagleSize
      - IFS: LocalFoodSize
      - MFE: MaxFunctionEvals
    """
    xMin = np.asarray(SpaceXMin, dtype=float).reshape(D)
    xMax = np.asarray(SpaceXMax, dtype=float).reshape(D)
    cfg = PeoaConfig(
        InitEagleSize=IES,
        LocalFoodSize=IFS,
        MaxFunctionEvals=MFE,
        Seed=Seed
    )
    optimizer = PhilippineEagleOptimizer(dimension=D, bounds=(xMin, xMax), objective=f, config=cfg)
    return optimizer.Run()
