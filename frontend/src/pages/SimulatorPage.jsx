import { useLocation, useNavigate } from "react-router-dom";
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

  const { pitcher, batter, outs, offScore, defScore, inning, bases } = state;

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

            <div className="scoreboard-row">
              <div className="label">Coming soon</div>
            </div>

            <div className="scoreboard-row">
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
          <div className="panel">Center coming soon</div>
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
