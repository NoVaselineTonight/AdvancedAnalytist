"""Joint extrapolation (J3): where is a player who is not seen, given everybody else?

Each track alone only knows its own last sightings. The rest of the court knows more:
  * the play moves together - when the players around a hidden player (and the ball) run down the court and back,
    the hidden player almost certainly did too. ``JointField.residual`` is the distance-weighted mean deviation of
    the other players' (and the ball's) motion from their own straight line over the same time span (bridges) or
    their displacement since the last sighting (open end). Added to the single-track estimate with gain alpha.
  * camera view: a player who is NOT detected while the camera sees his predicted spot (not behind anyone) is
    not there - ``exclude_view`` moves such predictions to the nearest spot that is out of view or hidden.
  * court bounds: predictions stay on the court (+ margin).
  * honest uncertainty: sigma grows with time since the last sighting (calibrated on hide-and-predict, see below).

Hide-and-predict on real tracks (seen samples of output tracks hidden; params chosen on b3min+p4400, numbers from
the held-out clips c300+c900; median / p90 error in m):
  bridge (both ends known), single-track smoother -> joint:  1 s 0.08/0.21 -> 0.08/0.22,  4 s 0.42/1.26 -> 0.42/1.28,
                                                              6 s 0.54/2.18 -> 0.56/1.93, 10 s 1.37/5.20 -> 1.17/4.23
  open end (last seen at a), velocity decay -> joint:        1 s 0.21/0.75 -> 0.20/0.71,  2 s 0.47/1.86 -> 0.43/1.54,
                                                              4 s 1.00/3.76 -> 0.93/3.11, 10 s 2.59/9.43 -> 2.09/7.30
  open-end 2-sigma coverage 93-100 %. ~1 ms per bridge + open prediction (vectorised over identities).
  Real re-sightings (same id unseen 3-6 s, n=131): decay 3.73/9.96 -> joint 3.53/8.50 m.
  Camera-view exclusion was NOT better on real re-sightings (3.53 -> 3.75 m median; 32 % of predictions moved -
  many 'unseen' players are missed detections in view), so callers should use it only where detection is
  reliable; not used by default.
Interaction priors (same test, 4 clips x 3 seeds, open end, median/p90 m vs plain joint): defender shadow
  4 s 0.81/2.59 -> 0.83/2.62 (=), 6 s 1.31/3.40 -> 1.17/3.26 and 1.11/2.81 -> 1.01/3.25, 10 s 1.34/3.40 -> 0.94/3.50 and
  2.01/4.51 -> 1.78/4.24; held-out c300+c900 alone 6 s 1.45/4.45 -> 1.10/3.43, 10 s 1.62/4.09 -> 1.26/2.97 -> kept as
  an option (extrapolate(..., defending_basket=)), median better at 6-10 s, p90 mixed. Offence spacing (team centroid
  offset) and transition drift (possession change -> towards the new half) were WORSE than plain joint at 4-10 s
  (e.g. spacing 10 s 1.80 -> 2.31 m, transition 6 s 1.03 -> 1.33 m) -> dropped. Bridges: no prior helps.
Not wired into pass 2 (gaps the pipeline bridges are mostly < 5 s, where it gives nothing). API for the joint
tracker / lineup slots (J1/J2): JointField.from_rows(...).bridge(...) / .extrapolate(...).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA_BRIDGE = 0.3
ALPHA_OPEN = 0.6
NEIGHBOUR_L_M = 6.0  # weight exp(-distance at the last sighting / L)
BALL_WEIGHT = 0.5
MIN_COVER = 0.6  # a reference player must be seen in >= this share of the span
OPEN_TAU_S = 1.0  # own velocity decays with this time constant (open end)
OPEN_SIGMA_M_PER_SQRT_S = 1.5  # open-end spread (hide-and-predict: 2 sigma covers ~95 % up to 10 s)
OPEN_RAMP_S = 3.0
SHADOW_MAX_M = 4.0  # defender shadow: nearest opponent at most this far at the last sighting
SHADOW_TAU_S = 3.0
SHADOW_GAP_M = 1.5  # defender stands this far from his man, towards the defended basket
BRIDGE_MIN_S = 5.0  # shorter bridges: the single-track smoother is as good (hide-and-predict)


@dataclass
class JointField:
    """Seen positions of every identity per sample (nan = not seen) + ball positions."""

    t: np.ndarray
    P: dict  # id -> (n, 2)
    team: dict  # id -> team code
    ball: np.ndarray  # (n, 2), nan = unknown

    @classmethod
    def from_rows(cls, rows, t: np.ndarray, idx_of_k: dict[int, int], ball_xy: np.ndarray | None = None
                  ) -> "JointField":
        """rows: export.TrackRow list (players/referees); a row counts as seen when it has a box (measured)."""
        n = len(t)
        P, team = {}, {}
        for r in rows:
            if r.object_type == "ball" or r.box is None:
                continue
            i = idx_of_k.get(int(r.k))
            if i is None:
                continue
            a = P.get(r.track_id)
            if a is None:
                a = P[r.track_id] = np.full((n, 2), np.nan)
                team[r.track_id] = r.team
            a[i] = (r.x, r.y)
        ball = np.full((n, 2), np.nan) if ball_xy is None else np.asarray(ball_xy, float)
        return cls(np.asarray(t, float), P, team, ball)

    def _stack(self):
        st = getattr(self, "_st", None)
        if st is None:
            ids = list(self.P)
            A = np.stack([self.P[q] for q in ids] + [self.ball]) if ids else self.ball[None]
            w = np.r_[np.ones(len(ids)), BALL_WEIGHT]
            st = self._st = (ids, A, w)
        return st

    def residual(self, tid, a: int, ks: np.ndarray, p0: np.ndarray, b: int | None = None) -> np.ndarray:
        """Common motion of the neighbours at samples ks (a < ks): deviation from their straight line a->b, or
        (b None) their displacement since a. Zero where nobody informative is seen. Vectorised over identities."""
        ks = np.asarray(ks, int)
        ids, A, wq = self._stack()
        use = np.isfinite(A[:, a]).all(axis=1)
        if b is not None:
            use &= np.isfinite(A[:, b]).all(axis=1)
        if tid in self.P:
            use[ids.index(tid)] = False
        if not use.any() or not len(ks):
            return np.zeros((len(ks), 2))
        Q = A[use]
        seg = Q[:, ks]  # (m, L, 2)
        ok = np.isfinite(seg).all(axis=2)
        cov = ok.mean(axis=1) >= MIN_COVER
        if not cov.any():
            return np.zeros((len(ks), 2))
        Q, seg, ok = Q[cov], seg[cov], ok[cov]
        w = wq[use][cov] * np.exp(-np.hypot(*(Q[:, a] - p0).T) / NEIGHBOUR_L_M)
        if b is None:
            dev = seg - Q[:, a][:, None]
        else:
            f = ((ks - a) / max(1, b - a))[None, :, None]
            dev = seg - (Q[:, a][:, None] + f * (Q[:, b] - Q[:, a])[:, None])
        dev = np.where(ok[..., None], dev, 0.0)
        W = w[:, None] * ok
        den = W.sum(axis=0)
        num = (W[..., None] * dev).sum(axis=0)
        return np.where(den[:, None] > 0, num / np.maximum(den, 1e-9)[:, None], 0.0)

    def bridge(self, tid, a: int, b: int, ks: np.ndarray, base_xy: np.ndarray, p_a: np.ndarray) -> np.ndarray:
        """Single-track bridge (smoother output at ks, a < ks < b; p_a = position at a) + alpha x the neighbours'
        common motion."""
        return base_xy + ALPHA_BRIDGE * self.residual(tid, a, ks, p_a, b)

    def extrapolate(self, tid, a: int, ks: np.ndarray, p_a: np.ndarray, v_a: np.ndarray,
                    defending_basket: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Open end: last seen at sample a (position p_a, velocity v_a m/s). Returns (xy, sigma) at ks > a.

        defending_basket (court xy of the basket the player's team defends, i.e. the opponents are in possession -
        from game_state possessions + attack direction): defender-shadow prior - the nearest opponent within
        SHADOW_MAX_M at a is followed with the offset at a, which relaxes (SHADOW_TAU_S) to SHADOW_GAP_M on his
        basket side; the result is averaged 50/50 with the plain joint prediction."""
        ks = np.asarray(ks, int)
        dt = (self.t[ks] - self.t[a])[:, None]
        own = p_a + v_a * OPEN_TAU_S * (1 - np.exp(-dt / OPEN_TAU_S))
        ramp = np.minimum(1.0, dt / OPEN_RAMP_S)  # own velocity first, the play's motion takes over
        xy = own + ALPHA_OPEN * ramp * self.residual(tid, a, ks, p_a)
        if defending_basket is not None:
            sh = self._shadow(tid, a, ks, p_a, np.asarray(defending_basket, float), dt)
            if sh is not None:
                xy = 0.5 * xy + 0.5 * np.where(np.isfinite(sh), sh, xy)
        sigma = 0.3 + OPEN_SIGMA_M_PER_SQRT_S * np.sqrt(dt[:, 0])
        return clip_court(xy), sigma

    def _shadow(self, tid, a, ks, p_a, basket, dt):
        T = self.team.get(tid)
        opp = [q for q, Q in self.P.items() if q != tid and self.team.get(q) not in (None, T, "referee", "unknown")
               and self.team.get(q) in ("A", "B") and np.isfinite(Q[a]).all()]
        if not opp:
            return None
        q = min(opp, key=lambda q: float(np.hypot(*(self.P[q][a] - p_a))))
        if np.hypot(*(self.P[q][a] - p_a)) > SHADOW_MAX_M:
            return None
        Q = self.P[q]
        idx = np.flatnonzero(np.isfinite(Q[:, 0]))
        Qk = np.c_[np.interp(ks, idx, Q[idx, 0]), np.interp(ks, idx, Q[idx, 1])]
        Qk[ks > idx[-1]] = np.nan  # never extrapolate the man himself
        d = basket - Qk
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        lam = 1 - np.exp(-dt / SHADOW_TAU_S)
        return Qk + (1 - lam) * (p_a - Q[a]) + lam * SHADOW_GAP_M * d


def clip_court(xy: np.ndarray, margin: float = 1.0) -> np.ndarray:
    import replay.court as court

    out = np.array(xy, float)
    out[:, 0] = np.clip(out[:, 0], -margin, court.LENGTH + margin)
    out[:, 1] = np.clip(out[:, 1], -margin, court.WIDTH + margin)
    return out


_RING = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)])


def exclude_view(ctx, ks: np.ndarray, xy: np.ndarray, max_m: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera-view exclusion: a prediction whose spot is VISIBLE (in the picture, nobody in front) while the
    player was not detected is moved to the nearest point (rings of 0.5 m up to max_m) that is out of view or
    hidden behind someone. ctx = court_tracker.OcclusionContext. Returns (xy, moved mask)."""
    from replay.court_tracker import VISIBLE

    out = np.array(xy, float)
    moved = np.zeros(len(out), bool)
    codes = ctx.classify_many(ks, out)
    for j in np.flatnonzero(codes == VISIBLE):
        i = int(ks[j])
        for r in np.arange(0.5, max_m + 1e-9, 0.5):
            cand = clip_court(out[j] + r * _RING)
            c = ctx.classify_many(np.full(len(cand), i), cand)
            good = np.flatnonzero(c != VISIBLE)
            if len(good):
                out[j] = cand[good[0]]
                moved[j] = True
                break
    return out, moved
