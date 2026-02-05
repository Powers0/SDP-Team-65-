import { useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import Select from "react-select";
import { getPlayers } from "../api";
import { useLocation } from "react-router-dom";
import "../HomePage.css";

export default function HomePage() {
  const navigate = useNavigate();

  const [players, setPlayers] = useState({ pitchers: [], batters: [] });
  const [pitcher, setPitcher] = useState(null);
  const [batter, setBatter] = useState(null);
  const [err, setErr] = useState("");

  // Game context state
  const [outs, setOuts] = useState(0);
  const [offScore, setOffScore] = useState(0);
  const [defScore, setDefScore] = useState(0);
  const [inning, setInning] = useState(1);
  const [bases, setBases] = useState({ on1: false, on2: false, on3: false });
  const { state } = useLocation();

  // black text, readable options
  const selectStyles = {
    control: (base, state) => ({
      ...base,
      minHeight: 36,
      height: 36,
      borderRadius: 10,
      borderColor: state.isFocused ? "#999" : "#ccc",
      boxShadow: state.isFocused ? "0 0 0 1px #999" : "none",
      paddingLeft: 2,
    }),
    valueContainer: (base) => ({
      ...base,
      height: 36,
      padding: "0 10px",
    }),
    input: (base) => ({
      ...base,
      margin: 0,
      padding: 0,
      color: "black",
    }),
    singleValue: (base) => ({
      ...base,
      color: "black",
      fontSize: 14,
    }),
    placeholder: (base) => ({
      ...base,
      color: "#666",
      fontSize: 14,
    }),
    indicatorsContainer: (base) => ({
      ...base,
      height: 36,
    }),
    dropdownIndicator: (base) => ({
      ...base,
      padding: 6,
    }),
    clearIndicator: (base) => ({
      ...base,
      padding: 6,
    }),
    option: (base, state) => ({
      ...base,
      color: "black",
      backgroundColor: state.isFocused ? "#eee" : "white",
      fontSize: 14,
      paddingTop: 10,
      paddingBottom: 10,
    }),
    menu: (base) => ({
      ...base,
      borderRadius: 10,
      overflow: "hidden",
    }),
  };

  function setDefaults() {
    setPitcher(null);
    setBatter(null);

    setOuts(0);
    setOffScore(0);
    setDefScore(0);
    setInning(1);
    setBases({ on1: false, on2: false, on3: false });
  }

  useEffect(() => {
    (async () => {
      try {
        const data = await getPlayers();
        setPlayers(data);

        if (state?.mode === "preserve") {
          setPitcher(state.pitcher);
          setBatter(state.batter);
          setOuts(state.outs);
          setOffScore(state.offScore);
          setDefScore(state.defScore);
          setInning(state.inning);
          setBases(state.bases);
        } else {
          // covers both first time load state and reset mode
          setDefaults(data);
        }
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, [state]);

  const pitcherOptions = useMemo(
    () => players.pitchers.map((p) => ({ value: p.id, label: p.label })),
    [players.pitchers],
  );

  const batterOptions = useMemo(
    () => players.batters.map((b) => ({ value: b.id, label: b.label })),
    [players.batters],
  );

  const canPlay = Boolean(pitcher && batter);

  //  clamp score and inning
  function clampScore(v) {
    const n = Number(v);
    if (Number.isNaN(n)) return 0;
    return Math.max(0, Math.min(99, n));
  }

  function clampInning(v) {
    const n = Number(v);
    if (Number.isNaN(n)) return 1;
    return Math.max(1, Math.min(9, n));
  }

  return (
    <div className="home">
      <div className="home-card">
        <div className="home-header">
          <h1>At-Bat Simulator</h1>
          <p>Simulate MLB at-bats based on game context</p>
        </div>

        {err && <p style={{ color: "crimson" }}>{err}</p>}

        <div className="player-select-section">
          <div className="player-select-row">
            <div className="player-select">
              <label>Select MLB Pitcher</label>
              <Select
                value={pitcher}
                onChange={setPitcher}
                options={pitcherOptions}
                placeholder="Search pitcher..."
                isClearable
                isSearchable
                styles={selectStyles}
              />
            </div>

            <div className="player-select">
              <label>Select MLB Batter</label>
              <Select
                value={batter}
                onChange={setBatter}
                options={batterOptions}
                placeholder="Search batter..."
                isClearable
                isSearchable
                styles={selectStyles}
              />
            </div>
          </div>
        </div>

        <div className="game-context">
          <div className="game-context-header">
            <h2>Game Context</h2>
          </div>

          <div className="game-context-body">
            {/* OUTS */}
            <div className="context-field">
              <div className="context-label-row">
                <label className="context-label">Outs:</label>

                <div className="outs-lights" aria-label="Outs">
                  {[1, 2].map((n) => (
                    <label
                      key={n}
                      className="outs-option"
                      title={`${n} out${n === 1 ? "" : "s"}`}
                    >
                      <input
                        type="checkbox"
                        checked={outs >= n}
                        onChange={() => {
                          if (n === 1) setOuts(outs >= 1 ? 0 : 1);
                          else setOuts(outs === 2 ? 1 : 2);
                        }}
                        aria-label={`${n} out${n === 1 ? "" : "s"}`}
                      />
                      <span className="outs-light" aria-hidden="true" />
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* SCORE */}
            <div className="context-field">
              <div className="context-label-row">
                <label className="context-label">Score:</label>

                <div className="score-inputs" aria-label="Score">
                  <input
                    className="score-box"
                    type="number"
                    min="0"
                    max="99"
                    value={offScore}
                    onChange={(e) => setOffScore(clampScore(e.target.value))}
                    aria-label="Offensive Score"
                  />
                  <span className="score-dash">–</span>
                  <input
                    className="score-box"
                    type="number"
                    min="0"
                    max="99"
                    value={defScore}
                    onChange={(e) => setDefScore(clampScore(e.target.value))}
                    aria-label="Defensive Score"
                  />
                </div>
              </div>
              <small className="score-hint">(Off-Def)</small>
            </div>

            <div className="context-field">
              <div className="context-label-row">
                <label className="context-label">Baserunners:</label>
                <div className="bases-diamond">
                  <button
                    type="button"
                    className={`base base-3b ${bases.on3 ? "on" : ""}`}
                    onClick={() => setBases((p) => ({ ...p, on3: !p.on3 }))}
                    aria-pressed={bases.on3}
                    aria-label="Runner on 3B"
                  />
                  <button
                    type="button"
                    className={`base base-2b ${bases.on2 ? "on" : ""}`}
                    onClick={() => setBases((p) => ({ ...p, on2: !p.on2 }))}
                    aria-pressed={bases.on2}
                    aria-label="Runner on 2B"
                  />
                  <button
                    type="button"
                    className={`base base-1b ${bases.on1 ? "on" : ""}`}
                    onClick={() => setBases((p) => ({ ...p, on1: !p.on1 }))}
                    aria-pressed={bases.on1}
                    aria-label="Runner on 1B"
                  />
                </div>
              </div>
            </div>

            <div className="context-field">
              <div className="context-label-row">
                <label className="context-label">Inning:</label>
                <input
                  className="inning-box"
                  type="number"
                  min="1"
                  max="9"
                  value={inning}
                  onChange={(e) => setInning(clampInning(e.target.value))}
                  aria-label="Inning"
                />
              </div>
            </div>
          </div>
        </div>

        <button
          className={`play-button ${canPlay ? "" : "disabled"}`}
          disabled={!canPlay}
          onClick={() => {
            if (!canPlay) return;

            navigate("/simulator", {
              state: {
                pitcher,
                batter,
                outs,
                offScore,
                defScore,
                inning,
                bases,
              },
            });
          }}
        >
          Play Ball!
        </button>
      </div>
    </div>
  );
}
