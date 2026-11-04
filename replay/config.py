"""All processing tunables in one place.

Defaults are chosen for 720p-1080p sideline footage at a 10 fps processing rate.
Override any field with a JSON file passed via ``--config``; only the keys you
set are changed, e.g. ``{"process_fps": 5, "detector": {"variant": "nano"}}``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectorConfig:
    backend: str = "auto"  # auto | onnx | torch.  auto = ONNX Runtime on the GPU when available, else PyTorch CPU
    device: str = "auto"  # auto | gpu | cpu
    variant: str = "medium"  # nano | small | medium | large
    threshold: float = 0.30  # keep low; the tracker uses low-score boxes too (ByteTrack)
    tiles: tuple[int, int] = (1, 1)  # (rows, cols); >1 helps when players are tiny in frame
    tile_overlap: float = 0.15
    resolution: int | None = None  # None = model default
    # Task 9 (TRACKING_FIX_PLAN): also store PERSON boxes scoring in [store_threshold, threshold) so ByteTrack's
    # second association and pass 2 can extend identities through occlusions. None = off (default); turning it on
    # changes the run signature (pipeline_version 2). Pass 2 never lets such a row start an identity, enter team
    # colours, jersey reading, re-ID or the roster reference crowd (conf < threshold filters).
    store_threshold: float | None = None


@dataclass
class TrackingConfig:
    activation_threshold: float = 0.45
    lost_track_buffer_s: float = 2.0
    matching_threshold: float = 0.8
    min_consecutive_frames: int = 2
    camera_motion_compensation: bool = True
    # Task 7 (TRACKING_FIX_PLAN): Kalman noise scaled from supervision's 30-fps tuning to the sample rate, and
    # IoU matching on boxes expanded by this fraction of the box height. OFF by default (changes the run signature).
    kalman_fps_scale: bool = False
    box_expand: float = 0.0


@dataclass
class MotionConfig:
    """Frame-to-frame camera motion estimation (for panning cameras)."""

    enabled: bool = True  # False = camera is fixed; one calibration is valid for every frame
    work_width: int = 640
    max_corners: int = 800
    min_inliers: int = 40
    ransac_px: float = 2.0  # at work_width scale
    # Task 6: when the estimate fails, the tracker re-uses the last good per-sample motion for at most this many
    # consecutive samples (0 = off, the default; the stored motion stays None either way)
    fallback_samples: int = 0


@dataclass
class CalibrationConfig:
    max_propagation_s: float = 600.0  # frames further than this from any keyframe get no calibration
    keyframe_snap_s: float = 0.2
    max_reprojection_px: float = 10.0  # keyframes worse than this are rejected
    # trained court detector (replay/courtnet.py, autocourt.calibrate_net): "off" | "fallback" (only when the classic
    # method finds too little) | "first" (detector first, classic as fallback). Env REPLAY_COURT_DETECTOR overrides.
    court_detector: str = "off"


@dataclass
class FilterConfig:
    # hard cut before tracking, deliberately generous: calibration errors put players at the baseline / an
    # inbounding player / referees on the lines 2-3 m "outside" (with 2.0 m, ~2 % of the visible players and
    # ~15 % of the referees were lost on the test clips). roster.py's soft court cue and the per-frame limits
    # handle the people further out.
    court_margin_m: float = 3.5
    drop_cut_feet: bool = True  # boxes touching the bottom edge / an overlay have no visible feet
    min_track_frames: int = 5
    # Task 4: a too-short identity is kept when it continues a kept identity of the same kit within this window
    # (before or after) at a reachable distance (roster.short_but_continuing): short re-appearances between occlusions
    short_keep_window_s: float = 0.0  # 0 = off. 3.0 gated 18.10 (t4-short-s3): ID switches 40 -> 28 better, recall/
    # phantoms/transfers within noise, but out IDs/min 71 -> 82 and ID life 16 -> 11 s WORSE (kept crumbs = own ids)
    short_keep_reach_mps: float = 7.0
    # Player vs bench/coaches/officials/spectators (replay/roster.py). Each cue gives evidence in [-1, 1]
    # (+ = player); score = prior + sum(weight * evidence), smoothed along the track, kept when > 0.
    prior: float = 0.5
    # static: share of time the spot under the feet is occupied (camera-stabilised image space)
    static_window_s: float = 45.0  # +- this around each detection
    static_sample_s: float = 0.5
    static_min_dt_s: float = 1.5  # ignore frames this close (everyone is "static" over a fraction of a second)
    static_radius_rel: float = 0.45  # same spot = foot points within this x box height
    static_height_ratio: float = 2.0  # occupant's box height within this factor (seated boxes fluctuate)
    static_color_veto: bool = True  # with jersey colours: the occupant must look alike (realign_max_color)
    static_min_visible_s: float = 3.0  # spot must be in view at least this long to judge
    static_full_s: float = 12.0  # evidence counts fully once the spot was seen this long
    # chained camera motion drifts ~4 % of the camera's travel (pan away and back = 100+ px), so only frames
    # within this much camera travel are compared (same "visit" of that part of the gym)
    static_max_travel_px: float = 900.0
    static_occupancy: float = 0.62  # occupancy where the evidence is neutral
    static_occupancy_scale: float = 0.15
    w_static: float = 2.0
    # chained camera motion drifts across fast pans; re-register each frame pair by the dominant shift of
    # the people in it (static people agree), allowing up to 20 px + this share of the camera's travel
    realign_drift_rel: float = 0.03  # measured: ~1-2 % of the camera's travel after pans away and back
    realign_min_support: int = 3  # people that must agree on the shift
    realign_max_color: float = 25.0  # torso Lab distance above which two boxes can't be the same person
    # activity: 90th percentile wander of the track around its median position, in box heights
    activity_window_s: float = 20.0
    activity_min_s: float = 4.0
    activity_full_s: float = 20.0  # stillness counts fully once the track was watched this long
    active_spread: float = 0.6  # neutral
    active_spread_scale: float = 0.4
    w_activity: float = 1.2
    # posture
    seated_aspect: float = 1.6  # box height/width below this = sitting (bench, spectators)
    seated_rel_height: float = 0.78  # box height / standing height at the same image row
    w_posture: float = 1.0
    # court position (only with calibration; soft because calibrations are off by 15-40 px)
    off_court_m: float = 1.0  # neutral distance outside the lines
    off_court_scale_m: float = 1.0
    w_court: float = 1.5
    court_inside_weight: float = 0.3
    # Standing still is bench evidence only OUTSIDE the lines: players lined up along the lane at free throws,
    # players/referees waiting during a dead ball stand still for 10-60 s inside the court. Negative
    # static/activity evidence fades from full at deep_inside_m (metres inside the nearest line) to
    # deep_inside_min_weight at deep_inside_full_m. Seated people inside (calibration error) are still removed
    # by the posture cue. (Measured: free-throw lineups at 112-125 s of clip-4400 fully drawn; referees too.)
    deep_inside_m: float = 0.3
    deep_inside_full_m: float = 1.0
    deep_inside_min_weight: float = 0.0
    # per-frame limits (max_on_court here, max_players / max_players_shown in pass 2) keep the most player-like
    # people: roster score + this bonus per metre inside the nearest line, up to cap_inside_full_m
    cap_inside_bonus: float = 1.0
    cap_inside_full_m: float = 3.0
    decision_smooth_s: float = 6.0  # score averaged along the track over +- this (decides per time window)
    max_on_court: int = 13  # 10 players + 3 referees: keep at most this many per frame, most player-like first
    max_players: int = 12  # at most this many team players per frame (10 on court + slack for substitutions);
    # deliberately not per team: if team colours are mixed up, a per-team cap would delete real players
    max_referees: int = 3
    clip_bench: bool = False  # gym: CLIP per-game probe drops kept tracks that look like bench / coaches (clip_roles);
    # gate cb-on-s3 vs cb-off-s3 (c300,c900,b3min): phantoms 0.68->0.61, IDs/min 70.7->68.0, life 14.5->15.0 better,
    # extrap 5 s 2.21->2.31 WORSE; +25-30 s CLIP embedding per 2-min clip on the first build (cached)
    max_players_shown: int = 10  # after gap filling: never more players than can be on court
    # final limits per track segment (limits.py): an eviction shorter than this would cut a hole into a track -> the
    # next candidate goes instead; off-court runs shorter than offcourt_run_s inside a track are foot-point noise
    segment_limits: bool = True  # False = the old per-frame limits (roster.limit_shown_rows + consolidate_drop)
    segment_rules: bool = False  # True = limits.apply_segment_limits (whole-run caps/ghosts); False = smoothed per-frame
    min_evict_s: float = 1.0
    offcourt_run_s: float = 0.5
    sticky_cap: bool = False  # caps with memory + fill-aware ranking (limits._sticky_caps, identity_design.md step E);
    # env REPLAY_STICKY_CAP=0/1 overrides. Gate e-on vs e-off (23.10) decides the default


@dataclass
class CourtTrackConfig:
    """Second-stage identity tracking (handles occlusion).

    method "offline" (default): tracklets + global linking over the whole video (tracklets.py);
    "online": the older frame-by-frame court tracker (court_tracker.py).
    """

    method: str = "offline"
    max_lost_s: float = 4.0  # online: an unseen player keeps their ID this long
    max_speed_mps: float = 8.0
    reach_slack_m: float = 1.0
    gate_chi2: float = 16.0  # online
    accel_noise: float = 4.0  # online
    lost_velocity_damping: float = 0.85  # online
    appearance_weight: float = 1.5  # online
    same_id_bonus: float = 3.0  # online
    new_track_min_conf: float = 0.35  # identities whose best detection scores below this are dropped
    # ---- offline (tracklets.py)
    max_speed_bh_s: float = 6.0  # running speed in camera-stabilised image space, body heights per second
    reach_slack_bh: float = 0.6
    end_window_s: float = 0.8  # tracklet end position/velocity from a line fit over this long
    height_window_s: float = 1.5  # full body height = 80th percentile of the box height within +- this
    full_height_percentile: float = 50.0  # higher picks up merged/oversized boxes as 'full height'
    partial_height_ratio: float = 0.8  # box shorter than this x full height = legs hidden -> foot re-estimated
    correct_partial_feet: bool = True  # re-estimate the foot of short boxes whose bottom is covered by someone
    partial_depth_scale: bool = True  # compare heights after removing the perspective size change (needs calibration)
    partial_foot_sigma_rel: float = 0.08  # foot uncertainty of a re-estimated foot, fraction of body height
    appearance_split: bool = True  # cut ByteTrack IDs where the jersey colour changes for good
    appearance_window_s: float = 1.0
    appearance_split_lab: float = 18.0  # a/b change (torso+shorts, Lab units) that counts as a different person
    occlusion_overlap: float = 0.2  # boxes overlapping more than this (of the smaller box) = possible ID swap
    chain_max_gap_s: float = 0.35  # detections without a ByteTrack ID are chained over gaps up to this
    max_overlap_samples: int = 3  # two pieces may overlap this many samples (duplicate boxes) and still join
    overlap_cost: float = 2.0
    bridge_velocity_s: float = 1.0  # constant-velocity bridging of a gap for at most this long
    pos_sigma_m: float = 0.5
    vel_sigma_mps: float = 1.5  # growth of position uncertainty during a gap
    img_sigma_bh: float = 0.2
    img_vel_sigma_bh: float = 0.9
    image_weight: float = 0.5  # weight of the image-space motion term when court positions exist too
    image_only_cost_per_s: float = 2.0  # extra gap cost when there is no court position (no calibration)
    max_height_log_ratio: float = 0.3
    height_weight: float = 0.5
    appearance_gate_chi2: float = 25.0  # robust: each dimension contributes at most 9
    link_appearance_weight: float = 1.5
    min_link_dets: int = 3  # pieces shorter than this are linked only over the shortest gaps (first round)
    gap_cost_per_s: float = 0.5
    absorb_max_dets: int = 8  # Task 4: 4 -> 8 (gate t4-absorb8-s3 18.10: all metrics within noise); leftover pieces this short may fill a gap of a longer identity (0 = off)
    absorb_window_s: float = 1.0  # ... if that identity is seen within this long before and after
    absorb_court_m: float = 1.0
    absorb_image_bh: float = 0.35
    use_reid: bool = False  # re-ID embedding cue in identity linking (tracklets.reid_enabled; env AA_IDENTITY_REID overrides)
    joint_tracker: bool = True  # gate r9-joint-s3 18.10: no WORSE, out ID life + extrap 3s better. J1 joint long-gap association (joint_tracker.py); env REPLAY_JOINT=0/1 overrides
    link_rounds_s: tuple[float, ...] = (0.4, 1.5, 4.0)  # max gap per linking round (longer gaps: too ambiguous)
    link_thresholds: tuple[float, ...] = (13.0, 18.0, 23.0)  # cost above which pieces stay unlinked


@dataclass
class BallConfig:
    """Ball tracking (ball_track.py: image-space trajectory linking; ball.py: court position)."""

    min_conf: float = 0.3  # candidates below this detector score are ignored (pass-1 detector threshold is 0.3)
    static_window_s: float = 3.0  # a "ball" that doesn't move for this long is a spare ball / false positive
    static_px: float = 6.0
    static_frac: float = 0.6  # ... in at least this share of the frames of +-static_window_s
    # trajectory linking (dynamic programming over all candidates, see ball_track.py)
    link_s: float = 1.0  # candidates up to this far apart in time can be linked (gap filled in the image)
    max_speed_mps: float = 16.0  # fastest plausible ball (image motion converted with the ball's own size)
    jitter_diam: float = 1.5  # detector / camera-motion jitter allowance, in ball diameters
    max_size_ratio: float = 2.5  # linked detections may differ in image size at most this much
    conf_offset: float = 0.2  # a candidate earns (conf - conf_offset)
    gap_penalty: float = 0.02  # per skipped frame inside a link
    motion_weight: float = 0.3  # cost of a link using the whole allowed displacement
    jump_penalty: float = 1.2  # teleport to an unrelated candidate within link_s (more than one detection earns)
    restart_penalty: float = 0.35  # new tracklet after more than link_s without a linked detection
    # court position (ball.py)
    holder_margin_w: float = 0.35  # ball centre may be this many box widths beside a player's box
    holder_ratio: float = 0.16  # ball image size / holder box height (measured median; 0.24 m ball, youth players)
    holder_ratio_tol: float = 2.0  # a ball more than this factor too big/small is at another depth: not held
    max_court_speed_mps: float = 14.0  # anchors implying a faster ball than this are dropped as outliers
    holder_keep_bonus: float = 0.35  # hysteresis: the current holder keeps the ball unless clearly beaten
    rim_radius_r: float = 2.5  # ball within this many rim radii of a rim (or above it) = at the basket
    max_flight_s: float = 3.0  # gaps between two anchors up to this long are filled (flight / hidden in hands)
    max_fill_s: float = 6.0  # gaps up to this long between two anchors are filled: holder kept, then the pass
    pass_speed_mps: float = 8.0  # assumed speed of a pass / shot when splitting a gap into "held" + "flight"
    hold_extend_s: float = 1.0  # unseen ball stays with its last / next holder this long at the ends of gaps
    same_holder_possessed_s: float = 2.0  # unseen between two sightings with the same holder: still possessed
    # holder decoding (ball.decode_holders) and free-throw prior (ball.free_throw_windows)
    holder_decode: bool = True  # Viterbi over all sightings (else greedy per sighting with hysteresis)
    free_cost: float = 1.0  # a sighting not held by anybody although someone's box could hold it
    free_switch_cost: float = 0.3  # holder <-> free (release / catch)
    holder_switch_cost: float = 0.2  # holder A -> holder B directly
    handoff_m: float = 1.6  # a direct change further than this needs a (missed) pass: + holder_jump_cost
    holder_jump_cost: float = 2.0
    jump_free_s: float = 0.6
    shot_soon_s: float = 2.0  # a holder change this long before the ball is at a rim costs no jump penalty  # ... fading out for sightings this far apart (the ball may have gone anywhere)
    slow_fill_holder: bool = False  # long no-holder stretches: ball at the player whose box holds its image point
    slow_fill_s: float = 1.5
    slow_fill_steady: float = 0.6  # ... the same player for at least this long
    ft_prior: bool = True  # free throws: only the shooter / referees hold the ball until the attempts are over
    ft_min_lane: int = 2
    ft_still_mps: float = 1.0
    ft_min_s: float = 7.0  # (a free-throw situation incl. set-up lasts 10-30 s; short look-alikes are live play)
    ft_gap_s: float = 1.5
    # ball at an untracked person's box -> their foot point. Off: mostly bench people, spectators, cut-off feet
    # (hand labels: teleports 4.0 -> 1.8 /min, median error 1.28 -> 1.02 m)
    person_anchors: bool = False


@dataclass
class ScoringConfig:
    """Automatic made-basket detection from the ball passing down through the rim (see scoring.py).

    Distances named ``*_r`` are in units of the rim's image radius (inner rim radius 0.225 m projected),
    so they scale with zoom. Tuned for precision over recall.
    """

    enabled: bool = True
    min_ball_conf: float = 0.3  # ball candidates below this detector score are ignored
    focal_range_rel: tuple[float, float] = (0.3, 15.0)  # plausible focal length / image width
    rim_smooth_s: float = 2.0  # running median of the rim position (camera-stabilised) over +-this
    rim_margin_px: float = 10.0  # rim must project at least this far inside the frame
    min_rim_px: float = 2.5  # rim radius below this many pixels = too far away to judge
    above_dy_r: tuple[float, float] = (-12.0, -0.4)  # ball centre above the rim centre (negative = up)
    above_dx_r: float = 4.0  # horizontal offset allowed while approaching
    below_dy_r: tuple[float, float] = (0.9, 7.0)  # ball centre below the rim (in the net / just under it)
    below_dx_r: float = 3.0  # = 1.5 rim widths
    max_fall_s: float = 1.2  # above -> below must happen within this time
    ball_size_ratio: tuple[float, float] = (0.45, 2.4)  # detected ball width / expected width at the rim
    static_window_s: float = 2.0  # candidates fixed relative to the rim for this long are not the ball
    static_frac: float = 0.5
    ballistic: bool = True  # fit a 3D parabola to the approach and veto clear misses (needs >= 4 points)
    ballistic_window_s: float = 1.5
    ballistic_max_rms_px: float = 4.0
    ballistic_veto_m: float = 0.8  # fitted rim-plane crossing further than this from the ring -> no basket
    debounce_s: float = 3.0  # at most one basket per rim within this time
    shooter_window_s: float = 4.0  # credited team = last possessor within this time before the shot
    min_confidence: float = 0.5  # events below this are dropped


@dataclass
class TeamConfig:
    """Jersey-colour team classification (teams.py, team_appearance.py)."""

    use_referee_class: bool = True
    clip_roles: bool = True  # broadcast footage: CLIP embeddings + per-game probe for referees (models/clip_b32_laion;
    # colour/zebra rules only when the model files are missing)
    clip_every_s: float = 0.5  # one embedded frame per this many seconds (pass 2, cached in work/clip_embeddings.npz)
    resample_from_video: bool = True  # pass 2 re-reads a sparse set of frames for jersey colour histograms (cached)
    resample_every_s: float = 0.3  # one decoded frame per this many seconds (all people in it are described)
    max_occlusion: float = 0.5  # sightings with more of the shirt hidden by someone in front are ignored
    n_clusters: int = 5  # two teams, referees, and room for bench / coaches / spectators
    same_hue_cos: float = 0.6  # two clusters with hue profiles more similar than this are the same colour
    min_team_colourful: float = 0.12  # share of clearly coloured pixels for a "coloured" team
    ref_min_achromatic: float = 0.35  # referee cluster: at least this share of grey/black/white pixels
    ref_max_colourful: float = 0.3
    ref_min_grey: float = 0.15  # a referee track needs this share of grey pixels (the shirt)
    fit_radius_scale: float = 1.15  # an "other" track joins a team if within this x the team's 90% spread
    min_fit_radius: float = 0.12  # Hellinger distance
    twin_hue_cos: float = 0.9  # a cluster with (nearly) the same distinctive hue as a team ...
    twin_min_rel: float = 0.85  # ... and at least this x its colourfulness is part of that team
    twin_min_rel_seeded: float = 0.75  # same with user-picked players (the anchored seed cluster stays small)
    team_mass_frac: float = 0.7  # ... or if it has this x the team's median amount of team-colour pixels
    user_color_max_cost: float = 60.0  # user kit colours must match the detected ones within this (hue degrees)
    # referee shirt test for a track's mean histogram (grey shirt over black trousers; 0 = off): rescues referees
    # that k-means put into a second achromatic cluster, keeps all-black staff out
    # roles by presence on court (teams = the two distinct kits with most people on court at a time, referee = next
    # distinct kit with <= ref_max_presence people); falls back to the colour rules when no second kit is on court
    role_by_presence: bool = True
    achromatic_kit_names: bool = True  # a team without a kit colour is shown/named white or black (lighter/darker)
    team_min_presence: float = 0.8  # people per frame
    team_min_sep: float = 0.25  # Hellinger distance between cluster histograms that counts as another kit
    team_max_spread: float = 0.33  # mean member distance to the cluster centre (sqrt-histogram units): kits are tight
    ref_max_spread: float = 0.38
    ref_max_presence: float = 3.5
    ref_min_presence: float = 0.2
    ref_shirt_min_grey: float = 0.10
    ref_shirt_max_colourful: float = 0.35
    ref_shirt_grey_black: float = 0.25  # grey share at least this x the black share
    ref_min_black: float = 0.14  # referees wear black trousers: a grey top over light/blue trousers is a spectator
    ref_max_sideline_out_m: float = 1.0  # a track whose median foot point is beyond a sideline by more is no referee
    cluster_court_weight: float = 0.0  # (OFF: unproven) k-means weight of a track ~ (1 - this + this * share of time inside the court)
    # a court track whose sightings show another kit for a sustained time is split there (identity switch)
    split_mixed_tracks: bool = True
    split_switch_cost: float = 10.0  # a label change costs this many disagreeing colour sightings
    split_min_evidence: int = 10  # each part needs this many colour sightings (one per ~0.3 s)
    # an "unknown" track joins a team when this share of its own colour sightings says that team (0 = off)
    adopt_vote_share: float = 0.4
    # sightings that look like nobody's kit (bench/other cluster) count as evidence for a split; off = only
    # orange <-> purple <-> referee changes split (dark kits seen side-on often look like "other")
    split_on_unknown: bool = False
    split_ref_frac: float = 0.4  # a split-off grey/black part is a referee when this share of it looks like one


@dataclass
class SmoothingConfig:
    """Court-position smoother (smoothing.py): robust RTS smoother, frame-rate independent."""

    max_gap_s: float = 3.0  # gaps up to this are interpolated (flagged); longer gaps are left empty
    reattach_hidden: bool = False  # court_tracker.reattach_hidden after linking (court_tracker.reattach_enabled; env REPLAY_REATTACH overrides)
    # Random acceleration of the motion model, m/s^2 per 0.1 s (continuous white-noise spectral density
    # accel_sigma^2 * 0.1 s). Higher = follows quick cuts more closely but keeps more jitter. (Replaces the
    # old per-sample "accel_noise", whose effect depended on the frame rate; old values are ignored.)
    accel_sigma: float = 2.0
    velocity_tau_s: float = 2.0  # velocity relaxes towards 0 with this time constant (0 = constant velocity);
    # keeps bridged gaps from overshooting
    foot_px_sigma_rel: float = 0.04  # foot-point pixel noise as a fraction of box height
    min_meas_sigma_m: float = 0.15
    gate_chi2: float = 16.0  # innovation gate (2 dof); measurements beyond are treated as outliers
    robust_nu: float = 4.0  # Student-t degrees of freedom for down-weighting surprising foot points
    robust_iters: int = 2  # reweighting passes against the smoothed track (0 = forward gating only)
    relock_s: float = 0.6  # this long of consistently "impossible" foot points = the player really is there
    # ---- occlusion-aware gap filling + uncertainty (smoothing.py / court_tracker.OcclusionContext)
    occlusion_aware: bool = True  # label filled samples occluded / off-screen (pos_state 4 / 5)
    occluded_max_gap_s: float = 6.0  # gaps up to this are bridged when the player was hidden (behind a nearer
    # person or out of view) along the way; 0 = off (only max_gap_s)
    occluded_gap_share: float = 0.7  # ... for at least this share of the gap samples
    max_speed_mps: float = 7.0  # a bridged gap must be coverable at this speed (+ 1 m)
    gap_sigma_m_per_sqrt_s: float = 1.5  # spread of a hidden player's position inside a gap (pos_sigma)
    snap_behind_occluder: bool = False  # undetected on a visible spot -> pulled behind the nearest person
    snap_sigma_m: float = 0.7  # softness of that pull
    game_aware_sigma: bool = False  # scale pos_sigma inside gaps by the game phase (dead ball 0.4x, transition 1.3x)
    capped_gap_max_s: float = 0.0  # gaps longer than max_gap_s up to this are filled only near their ends, where
    # pos_sigma <= sigma_cap_m (players who just left the picture stay a bit); 0 = off
    sigma_cap_m: float = 1.5


@dataclass
class FootConfig:
    """Robust foot points + per-detection position uncertainty (footpoint.py, pass-2 step 2)."""

    enabled: bool = False  # OFF by default until the benchmark A/B is clean (see data/resume/footpoint.md)
    keep_old_membership: bool = True  # only refine detections the plain box-bottom projection keeps on court
    max_shift_m: float = 2.0  # guard: repaired positions moving further than this from the plain one are not used
    airborne_skip_occluded: bool = True  # guard: no jump replacement when someone in front covers the box bottom
    hampel_skip_occluded: bool = False  # guard: no box-bottom repair when someone in front covers the box bottom
    # ---- airborne (jump) detection
    airborne: bool = True
    size_window_s: float = 1.5  # typical metric body size = rolling median over +- this
    jump_size_ratio: float = 1.12  # metric size this much above typical = candidate airborne / arms up
    jump_min_radial_m: float = 0.6  # ... and the position must move this far away from the camera (radially)
    jump_hysteresis: float = 0.35  # neighbouring samples above 1 + this x (ratio - 1) extend a jump
    jump_consistency: tuple[float, float] = (0.5, 2.0)  # observed / predicted radial displacement (camera model)
    jump_max_s: float = 1.2
    common_mode_min: int = 4  # >= this many people in a frame: divide out their common size change (calibration)  # airborne at most this long (longer = really moved there)
    airborne_sigma_m: float = 0.5  # extra uncertainty of an airborne sample's replaced position
    # ---- box-bottom glitches + cut boxes: foot from the box top + typical pixel height
    bottom_hampel: bool = True  # box bottom off the track's line while the top is on it -> bottom replaced
    hampel_min_px: float = 5.0
    hampel_rel: float = 0.08  # deviation threshold, fraction of box height
    glitch_px_factor: float = 2.0  # pixel noise multiplier for a replaced bottom
    fix_glitches: bool = True
    glitch_ratio: float = 0.82  # metric size below this x typical = legs cut off (box bottom wrong)
    glitch_max_s: float = 0.5
    recover_cut: bool = False  # (off until measured better) boxes cut by the image bottom / overlay: estimate the foot from the head
    head_size_sigma_rel: float = 0.08  # variability of the typical body size (crouch, arms, stride)
    max_head_sigma_m: float = 0.9  # head-based foot kept only if its 1-sigma position error is below this
    # ---- uncertainty
    foot_px_sigma_rel: float = 0.04  # pixel noise of a clean box bottom, fraction of box height
    min_px_sigma: float = 1.0
    min_sigma_m: float = 0.15
    occluded_px_factor: float = 1.0  # another box covers the bottom of this one (1.0 = no inflation; 1.8 untested)


@dataclass
class InPlayConfig:
    """Preliminary analysis (replay/inplay.py): find when the ball is in play so pass 1 skips breaks."""

    enabled: bool = True  # False = pass 1 analyses the whole video
    sample_fps: float = 2.0  # decode rate of the fast pre-pass
    analysis_width: int = 640  # decode width (grey)
    window_s: float = 20.0  # smoothing window of the activity signals
    pan_move: float = 0.004  # camera shift per sample (fraction of width) that counts as panning
    quiet_rel: float = 0.2  # "quiet" = floor activity below this x the video's typical in-play level
    quiet_min: float = 0.015
    quiet_max: float = 0.08
    quiet_pan_frac: float = 0.12  # ... and the camera pans in less than this share of samples
    hysteresis: float = 1.5  # a quiet stretch extends while activity < threshold x this
    min_break_s: float = 40.0  # quiet stretches shorter than this stay "in play" (free throws, slow play)
    still_break_s: float = 60.0  # camera not panning for this long = break (needs footage that usually pans)
    still_activity_rel: float = 0.3  # ... and median floor activity below this x the in-play level
    min_pan_level: float = 0.15  # ... i.e. typical pan share (75th percentile) at least this
    min_play_s: float = 10.0  # play shorter than this between two breaks is dropped
    pad_s: float = 5.0  # widen every in-play interval by this much on both sides
    merge_gap_s: float = 15.0  # breaks shorter than this (after padding) are analysed anyway
    use_clock: bool = True  # use a ticking scoreboard clock if one is found
    clock_stop_break_s: float = 45.0  # clock frozen (bar visible) at least this long = break
    clock_restart_slack_s: float = 10.0  # play may resume shortly before the operator restarts the clock
    clock_zero_level: float = 0.8  # block |diff| below this = unchanged (grey levels)
    clock_change_level: float = 4.0  # block |diff| above this = digit changed
    clock_min_digital: float = 0.9
    clock_min_tick_rate: float = 0.1
    clock_bar_tolerance: float = 10.0


# ---- moved here from redetect.py / reid.py (TRACKING_FIX_PLAN Task 0); re-exported there
@dataclass
class RedetectConfig:
    enabled: bool = False  # off by default until the benchmark confirms it (env REPLAY_REDETECT=1 turns it on)
    max_gap_s: float = 3.0  # only gaps shorter than this
    probe_every_s: float = 0.5  # at most one probe per this interval inside a gap
    crops_per_s: float = 1.0  # budget: crops per second of video
    threshold: float = 0.25  # detector score threshold inside crops (0.2 let bench people/partial boxes in)
    crop_scale: float = 2.6  # crop side = this x expected box height (gaps)
    min_crop_px: int = 128  # never crop smaller (limits the upscale to ~4.5x for a 576 px model input)
    min_track_samples: int = 3  # ignore ByteTrack tracks shorter than this
    max_hellinger: float = 0.42  # colour gate (same track median ~0.18, p95 ~0.41; other team p5 ~0.45)
    court_margin_m: float = 0.3  # gap anchors' feet must be on court (+margin)
    dup_iou: float = 0.5  # a new box overlapping an existing pass-1 box this much is the same person
    clusters: bool = False  # key-area cluster probes found ~2 people per 50 crops: budget goes to gaps/slots
    slots: bool = True  # probes where the previous pass-2 output predicted a hidden player (see hint_probes)
    max_hint_gap_s: float = 8.0  # hidden stretches longer than this are not probed (position too uncertain)
    key_margin_m: float = 1.0
    cluster_every_s: float = 1.0  # per basket side
    cluster_min_conf: float = 0.35
    gap_share: float = 0.9  # share of the budget reserved for gap probes (cluster probes found ~1 person per 18 crops)
    # Task 9 crumbs (low-score rows stored by pass 1 with DetectorConfig.store_threshold; none otherwise):
    # "fill" = kept only as gap fillers inside existing identities (redetect.fill_crumbs), "all", "none"
    crumbs: str = "fill"
    crumb_max_gap_s: float = 3.0
    crumb_max_lab_de: float = 20.0  # torso Lab distance to the gap's anchors (different ByteTrack id only)


@dataclass
class ReidConfig:
    enabled: bool = False  # opt-in: nothing in pass 2 depends on the model yet
    every_s: float = 0.3  # embed the person boxes of one sample every this many seconds
    min_box_h: int = 40  # px; smaller boxes are not embedded
    batch: int = 64
    device: str = "auto"  # "auto" (DirectML if available) | "cpu"
    model_path: str | None = None


@dataclass
class Config:
    # Analyse every Nth video frame (1 = every frame). Playback always shows every frame; court positions
    # are interpolated between analysed frames. When set, overrides process_fps.
    frame_stride: int | None = 3
    process_fps: float = 10.0  # used only when frame_stride is None
    chunk_frames: int = 300  # processed frames per resumable chunk
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    teams: TeamConfig = field(default_factory=TeamConfig)
    court_tracking: CourtTrackConfig = field(default_factory=CourtTrackConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    foot: FootConfig = field(default_factory=FootConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    inplay: InPlayConfig = field(default_factory=InPlayConfig)
    redetect: RedetectConfig = field(default_factory=RedetectConfig)  # pass 2 only (not in the run signature)
    reid: ReidConfig = field(default_factory=ReidConfig)  # pass 2 only (not in the run signature)
    # pass 2 only: court tracking + roster per game segment (split at long dead balls / breaks >= 20 s) in a process
    # pool (replay/parallel_pass2.py, SPEED_PLAN S4). Off until gated; env REPLAY_PARALLEL_P2=1/0 overrides.
    parallel_pass2: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_fps(self, video_fps: float) -> float:
        if self.frame_stride:
            return video_fps / max(1, int(self.frame_stride))
        return min(self.process_fps, video_fps)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        _merge(cfg, data)
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Keys written by older versions (e.g. an old results/config.used.json). Value = new name in the same section,
# or None = removed (its meaning changed or the feature is gone; the current default applies). Dropped silently.
_LEGACY_KEYS: dict[str, dict[str, str | None]] = {
    "SmoothingConfig": {"accel_noise": None},  # per-sample noise, replaced by the frame-rate independent accel_sigma
    "FilterConfig": {"min_in_court_frac": None, "min_moving_speed_mps": None},  # replaced by roster.py's cues
    "BallConfig": {"gate_px": None, "max_coast_s": None},
    "TeamConfig": {"unknown_distance": None, "min_vote_share": None},
}


def _merge(obj: Any, data: dict[str, Any]) -> None:
    known = {f.name: f for f in fields(obj)}
    legacy = _LEGACY_KEYS.get(type(obj).__name__, {})
    for key, value in data.items():
        if key not in known and key in legacy:
            key = legacy[key]
            if key is None or key in data:  # removed, or the new name is given as well (it wins)
                continue
        if key not in known:  # e.g. a section from an older version; ignore rather than refuse to load
            print(f"config: ignoring unknown key {key!r}")
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value)
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
