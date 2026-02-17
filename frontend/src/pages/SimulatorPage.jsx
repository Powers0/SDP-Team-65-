import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useRef, useState } from "react";
import "../SimulatorPage.css";

// Pitch type code -> human readable label
const PITCH_TYPE_NAMES = {
  FF: "4-Seam Fastball",
  FT: "2-Seam Fastball",
  SI: "Sinker",
  FC: "Cutter",
  SL: "Slider",
  CU: "Curveball",
  KC: "Knuckle Curve",
  CH: "Changeup",
  FS: "Splitter",
  SF: "Split-Finger",
  FO: "Forkball",
  SC: "Screwball",
  KN: "Knuckleball",
  EP: "Eephus",
  ST: "Sweeper",
};

function prettyPitchType(code) {
  if (!code) return "__";
  const c = String(code).toUpperCase().trim();
  return PITCH_TYPE_NAMES[c] ?? c;
}

// Strike-zone helpers
const PLATE_HALF_FT = 17 / 12 / 2; // 0.7083 ft

function isInStrikeZone(plateX, plateZ, szBot, szTop) {
  if (typeof plateX !== "number" || typeof plateZ !== "number") return false;
  if (typeof szBot !== "number" || typeof szTop !== "number") return false;
  const inX = plateX >= -PLATE_HALF_FT && plateX <= PLATE_HALF_FT;
  const inZ = plateZ >= szBot && plateZ <= szTop;
  return inX && inZ;
}

function computeCountFromPitches(pitches, upToIndexInclusive = null) {
  const end =
    upToIndexInclusive === null
      ? pitches.length
      : Math.max(0, Math.min(pitches.length, upToIndexInclusive + 1));

  let balls = 0;
  let strikes = 0;

  for (let i = 0; i < end; i += 1) {
    const r = String(pitches[i]?.result ?? "").toLowerCase();
    if (r.startsWith("ball")) balls += 1;
    if (r.startsWith("strike")) strikes += 1;
    if (balls > 3) balls = 3;
    if (strikes > 2) strikes = 2;
  }

  return { balls, strikes };
}

export default function SimulatorPage() {
  const { state } = useLocation();
  const navigate = useNavigate();

  // in case user refreshes or visits directly
  if (!state) {
    return (
      <div style={{ padding: 24 }}>
        <p>Missing game setup. Please go back and select players + context.</p>
        <button onClick={() => navigate("/")}>Back to Home</button>
      </div>
    );
  }

  const { pitcher, batter, outs, offScore, defScore, inning, bases } = state;

  useEffect(() => {
    const pitcherId =
      pitcher?.id ?? pitcher?.value ?? pitcher?.mlbam ?? pitcher?.mlbam_id;
    const batterId =
      batter?.id ?? batter?.value ?? batter?.mlbam ?? batter?.mlbam_id;

    console.groupCollapsed("[SIM] incoming state");
    console.log("pitcher:", pitcher);
    console.log("batter:", batter);
    console.log("pitcherId:", pitcherId, "batterId:", batterId);
    console.log("sz_bot/top:", batter?.sz_bot, batter?.sz_top);
    console.groupEnd();
  }, [pitcher, batter]);

  // pitch history
  const [pitches, setPitches] = useState([]); // each pitch: { plate_x, plate_z, pitchType, result }
  const [pitchIndex, setPitchIndex] = useState(-1); // -1 = "before pitch 1"
  const [contextLabel, setContextLabel] = useState(null); // e.g. "matchup", "pitcher+global"
  const pitchNum = pitchIndex >= 0 ? pitchIndex + 1 : "__";

  // current pitch (the one being viewed)
  const currentPitch = pitchIndex >= 0 ? pitches[pitchIndex] : null;

  // Count shown in the UI should reflect the pitch you're currently viewing.
  const displayCount = useMemo(() => {
    if (pitchIndex < 0) return { balls: 0, strikes: 0 };
    return computeCountFromPitches(pitches, pitchIndex);
  }, [pitches, pitchIndex]);

  // Count used for the next model call should reflect the *latest simulated state*.
  // (Important: if the user is viewing an old pitch via Prev Pitch, we still want the
  // next generated pitch to continue from the latest state.)
  const liveCount = useMemo(() => {
    return computeCountFromPitches(pitches, null);
  }, [pitches]);

  // --- MODEL API (Flask) ---
  // If your Flask server is running on a different host/port, update this.

  async function fetchModelPitch() {
    // Try to extract MLBAM ids from whatever shape you're currently passing.
    const pitcherId =
      pitcher?.id ?? pitcher?.value ?? pitcher?.mlbam ?? pitcher?.mlbam_id;
    const batterId =
      batter?.id ?? batter?.value ?? batter?.mlbam ?? batter?.mlbam_id;

    if (!pitcherId || !batterId) {
      console.warn("Missing pitcher/batter id for /api/predict", {
        pitcher,
        batter,
      });
      return null;
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pitcher_mlbam: Number(pitcherId),
          batter_mlbam: Number(batterId),
          user_context: {
            balls: liveCount?.balls ?? 0,
            strikes: liveCount?.strikes ?? 0,
            outs_when_up: outs ?? 0,
            on_1b: bases?.on1 ? 1 : 0,
            on_2b: bases?.on2 ? 1 : 0,
            on_3b: bases?.on3 ? 1 : 0,
            inning: inning ?? 1,
            score_diff: (offScore ?? 0) - (defScore ?? 0),
          },
        }),
      });

      const json = await res.json();
      console.log("Predict API response:", json);
      console.log("Predict API keys:", json ? Object.keys(json) : null);

      if (!res.ok) {
        console.warn("/api/predict returned non-OK", res.status, json);
        return null;
      }

      return json;
    } catch (e) {
      console.warn("/api/predict fetch failed", e);
      return null;
    }
  }

  // demo generator (replace later with API call)
  function makeDemoPitch() {
    return {
      plate_x: Math.random() * 3.6 - 1.8, // -1.8..1.8
      plate_z: 1.2 + Math.random() * 2.8, // 1.2..4.0
      pitchType: "FF",
      result: Math.random() > 0.5 ? "Strike!" : "Ball",
    };
  }

  async function onNextPitch() {
    // if we're not at the end of history, just move forward
    if (pitchIndex < pitches.length - 1) {
      setPitchIndex(pitchIndex + 1);
      return;
    }

    // Try model first (and log the returned JSON/keys)
    const api = await fetchModelPitch();

    if (api && api.context_label) {
      setContextLabel(api.context_label);
    }

    // Map API -> our pitch shape (use fallbacks so nothing breaks while you inspect keys)
    const mapped = api
      ? {
          plate_x:
            api.plate_x ??
            api.plateX ??
            api.px ??
            api.x ??
            api.location?.plate_x ??
            api.location?.x ??
            0,
          plate_z:
            api.plate_z ??
            api.plateZ ??
            api.pz ??
            api.z ??
            api.location?.plate_z ??
            api.location?.z ??
            2.6,
          pitchType:
            api.pitchType ??
            api.pitch_type ??
            api.pitch_type_code ??
            api.pitch ??
            api.pt ??
            "FF",
          // We derive Ball/Strike from the predicted location + hitter zone.
          // (Your API currently returns pitch_type + location; this keeps the sim consistent.)
          result: null,
        }
      : null;

    // If model didn't return something usable yet, fall back to demo
    const p = mapped ?? makeDemoPitch();

    // Derive Ball/Strike from the pitch location and the hitter's strike zone.
    // This is what advances the count for the next prediction.
    const px = Number(p?.plate_x);
    const pz = Number(p?.plate_z);
    const inZone = isInStrikeZone(px, pz, szBot, szTop);

    // Only overwrite if we don't already have a usable string.
    const existing = String(p?.result ?? "").trim();
    if (!existing) {
      p.result = inZone ? "Strike" : "Ball";
    }

    // Optional: store whether it was in-zone for debugging / UI later
    p.in_zone = inZone;

    setPitches((prev) => [...prev, p]);
    setPitchIndex((prev) => prev + 1);
  }

  function onPrevPitch() {
    setPitchIndex((i) => Math.max(-1, i - 1));
  }

  const zoneRef = useRef(null);
  const [zoneSize, setZoneSize] = useState({ w: 0, h: 0 });

  // World bounds (feet) for the whole box (includes balls)
  const X_MIN = -2.0;
  const X_MAX = 2.0;
  const Z_MIN = 0.5;
  const Z_MAX = 4.5;

  const PLATE_HALF = PLATE_HALF_FT; // keep existing math below unchanged

  useEffect(() => {
    if (!zoneRef.current) return;

    const el = zoneRef.current;

    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setZoneSize({ w: width, h: height });
    });

    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  function xToPx(x) {
    const pct = (x - X_MIN) / (X_MAX - X_MIN);
    return pct * zoneSize.w;
  }

  function zToPx(z) {
    const pct = 1 - (z - Z_MIN) / (Z_MAX - Z_MIN); // invert
    return pct * zoneSize.h;
  }

  const szBot = batter?.sz_bot ?? 1.5;
  const szTop = batter?.sz_top ?? 3.5;

  const zoneRect = useMemo(() => {
    const left = xToPx(-PLATE_HALF);
    const right = xToPx(PLATE_HALF);
    const top = zToPx(szTop);
    const bottom = zToPx(szBot);

    return {
      left,
      top,
      width: Math.max(0, right - left),
      height: Math.max(0, bottom - top),
    };
  }, [zoneSize.w, zoneSize.h, szBot, szTop]);

  const dot = useMemo(() => {
    if (!zoneSize.w || !zoneSize.h) return null;
    if (!currentPitch) return null;

    return {
      left: xToPx(currentPitch.plate_x),
      top: zToPx(currentPitch.plate_z),
    };
  }, [zoneSize.w, zoneSize.h, currentPitch]);

  useEffect(() => {
    console.log("batter keys:", batter ? Object.keys(batter) : null);
    console.log("batter full:", batter);
  }, [batter]);

  return (
    <div className="sim-page">
      <div className="sim-header">
        <button
          className="header-button"
          onClick={() =>
            navigate("/", {
              state: {
                pitcher,
                batter,
                outs,
                offScore,
                defScore,
                inning,
                bases,
                mode: "preserve",
              },
            })
          }
        >
          At-Bat Simulator
        </button>
        <div className="sim-divider" />
      </div>

      <div className="sim-main">
        {/* LEFT */}
        <div className="sim-left">
          <div className="panel scoreboard">
            <div className="scoreboard-header">
              <div className="scoreboard-hitter">
                <div className="label">Batter</div>
                <div className="value">
                  {batter?.label ?? batter?.name ?? "Unknown"}
                </div>
              </div>
            </div>

            <div className="scoreboard-row count">
              <div className="label">Count</div>

              <div className="count-display">
                <div className="count-lights" aria-label="Count lights">
                  <div className="count-row count-row-top">
                    <span
                      className={`count-dot ball ${displayCount.balls >= 1 ? "on" : ""}`}
                    />
                    <span
                      className={`count-dot ball ${displayCount.balls >= 2 ? "on" : ""}`}
                    />
                    <span
                      className={`count-dot ball ${displayCount.balls >= 3 ? "on" : ""}`}
                    />
                  </div>

                  <div className="count-row count-row-bottom">
                    <span
                      className={`count-dot strike ${displayCount.strikes >= 1 ? "on" : ""}`}
                    />
                    <span
                      className={`count-dot strike ${displayCount.strikes >= 2 ? "on" : ""}`}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="scoreboard-row outs">
              <div className="label">Outs</div>
              <div className="outs-lights" aria-label="Outs">
                <span className={`outs-light ${outs >= 1 ? "on" : ""}`} />
                <span className={`outs-light ${outs >= 2 ? "on" : ""}`} />
              </div>
            </div>

            <div className="scoreboard-row baserunners">
              <div className="label">BaseRunners</div>
              <div className="bases-diamond" aria-label="Baserunners">
                <span className={`base base-3b ${bases?.on3 ? "on" : ""}`} />
                <span className={`base base-2b ${bases?.on2 ? "on" : ""}`} />
                <span className={`base base-1b ${bases?.on1 ? "on" : ""}`} />
              </div>
            </div>
          </div>
        </div>

        {/* CENTER */}
        <div className="sim-center">
          <div className="zone-stack">
            {/* top pitcher */}
            <div className="pitcher-slot">{/* pitcher silhouette */}</div>

            {/* batter (left of zone) */}
            <div className="batter-slot">{/* batter silhouette */}</div>

            {/* zone */}
            <div className="zone-world" ref={zoneRef} aria-label="Strike zone">
              <div
                className="zone-rect"
                style={{
                  left: `${zoneRect.left}px`,
                  top: `${zoneRect.top}px`,
                  width: `${zoneRect.width}px`,
                  height: `${zoneRect.height}px`,
                }}
              />

              {dot && (
                <div
                  className="pitch-dot"
                  style={{ left: `${dot.left}px`, top: `${dot.top}px` }}
                />
              )}
            </div>

            {/* right of zone */}
            <div className="pitch-info">
              <div className="info-row">
                <span className="info-label">Pitch Type</span>
                <span className="info-value">
                  {prettyPitchType(currentPitch?.pitchType)}
                </span>
              </div>

              <div className="info-row">
                <span className="info-label">Pitch #</span>
                <span className="info-value">{pitchNum}</span>
              </div>
            </div>

            {/* under zone, BEFORE buttons */}
            <div className="pitch-result">
              {currentPitch ? currentPitch.result : "\u00A0"}
            </div>

            {/* buttons */}
            {/* buttons */}
            <div className="controls">
              <div className="controls-inner">
                {pitchIndex <= 0 ? (
                  // ON PITCH 0: Next Pitch + Pick New AB stacked
                  <div className="controls-p0">
                    <button className="btn primary" onClick={onNextPitch}>
                      Next Pitch
                    </button>

                    <button
                      className="btn ghost"
                      onClick={() => navigate("/", { replace: true })}
                    >
                      Pick New At-Bat
                    </button>
                  </div>
                ) : (
                  // ON PITCH 1+: Prev + Next row, Pick New AB centered under
                  <div className="controls-p1">
                    <div className="controls-top">
                      <button className="btn ghost" onClick={onPrevPitch}>
                        Prev Pitch
                      </button>
                      <button className="btn primary" onClick={onNextPitch}>
                        Next Pitch
                      </button>
                    </div>

                    <div className="controls-bottom">
                      <button
                        className="btn ghost"
                        onClick={() => navigate("/", { replace: true })}
                      >
                        Pick New At-Bat
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT */}
        <div className="sim-right">
          <div className="panel scoreboard">
            <div className="scoreboard-header">
              <div className="scoreboard-pitcher">
                <div className="label">Pitcher</div>
                <div className="value">
                  {pitcher?.label ?? pitcher?.name ?? "Unknown"}
                </div>
              </div>
            </div>

            <div className="scoreboard-row">
              <div className="label">Score</div>
              <div className="value mono">
                {offScore}–{defScore} <span className="hint">(Off–Def)</span>
              </div>
            </div>

            <div className="scoreboard-row">
              <div className="label">Inning</div>
              <div className="value mono">{inning}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
