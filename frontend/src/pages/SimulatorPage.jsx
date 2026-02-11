import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useRef, useState } from "react";
import "../SimulatorPage.css";

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

  const { pitcher, batter, outs, offScore, defScore, inning, bases, count } =
    state;

  // pitch history
  const [pitches, setPitches] = useState([]); // each pitch: { plate_x, plate_z, pitchType, result }
  const [pitchIndex, setPitchIndex] = useState(-1); // -1 = "before pitch 1"
  const pitchNum = pitchIndex >= 0 ? pitchIndex + 1 : "__";

  // current pitch (the one being viewed)
  const currentPitch = pitchIndex >= 0 ? pitches[pitchIndex] : null;

  // demo generator (replace later with API call)
  function makeDemoPitch() {
    return {
      plate_x: Math.random() * 3.6 - 1.8, // -1.8..1.8
      plate_z: 1.2 + Math.random() * 2.8, // 1.2..4.0
      pitchType: "FF",
      result: Math.random() > 0.5 ? "Strike!" : "Ball",
    };
  }

  function onNextPitch() {
    // if we're not at the end of history, just move forward
    if (pitchIndex < pitches.length - 1) {
      setPitchIndex(pitchIndex + 1);
      return;
    }

    // otherwise generate a new pitch and append
    const p = makeDemoPitch();
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

  const PLATE_HALF = 17 / 12 / 2; // 0.7083 ft

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
                      className={`count-dot ball ${count?.balls >= 1 ? "on" : ""}`}
                    />
                    <span
                      className={`count-dot ball ${count?.balls >= 2 ? "on" : ""}`}
                    />
                    <span
                      className={`count-dot ball ${count?.balls >= 3 ? "on" : ""}`}
                    />
                  </div>

                  <div className="count-row count-row-bottom">
                    <span
                      className={`count-dot strike ${count?.strikes >= 1 ? "on" : ""}`}
                    />
                    <span
                      className={`count-dot strike ${count?.strikes >= 2 ? "on" : ""}`}
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
                  {currentPitch?.pitchType ?? "__"}
                </span>
              </div>

              <div className="info-row">
                <span className="info-label">Location</span>
                <span className="info-value">
                  {currentPitch
                    ? `${currentPitch.plate_x.toFixed(2)}, ${currentPitch.plate_z.toFixed(2)}`
                    : "__"}
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
                      onClick={() =>
                        navigate("/", { state: { mode: "preserve" } })
                      }
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
                        onClick={() =>
                          navigate("/", { state: { mode: "preserve" } })
                        }
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
