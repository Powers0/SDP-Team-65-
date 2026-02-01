import { useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import Select from "react-select";
import { getPlayers } from "../api";
import "../HomePage.css";

export default function HomePage() {
  const navigate = useNavigate();

  const [players, setPlayers] = useState({ pitchers: [], batters: [] });
  const [pitcher, setPitcher] = useState(null);
  const [batter, setBatter] = useState(null);
  const [err, setErr] = useState("");
  const [outs, setOuts] = useState(0);

  // keep react-select styling intact (black text, readable options)
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

  useEffect(() => {
    (async () => {
      try {
        const data = await getPlayers();
        setPlayers(data);

        if (data.pitchers?.length) {
          setPitcher({
            value: data.pitchers[0].id,
            label: data.pitchers[0].label,
          });
        }
        if (data.batters?.length) {
          setBatter({
            value: data.batters[0].id,
            label: data.batters[0].label,
          });
        }
      } catch (e) {
        setErr(String(e));
      }
    })();
  }, []);

  const pitcherOptions = useMemo(
    () => players.pitchers.map((p) => ({ value: p.id, label: p.label })),
    [players.pitchers],
  );

  const batterOptions = useMemo(
    () => players.batters.map((b) => ({ value: b.id, label: b.label })),
    [players.batters],
  );

  return (
    <div className="home">
      <div className="home-card">
        <div className="home-header">
          <h1>At Bat Simulator</h1>
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
            <div className="context-field">
              <div className="context-label-row">
                <label className="context-label">Outs</label>
                <div
                  className="outs-lights"
                  role="radiogroup"
                  aria-label="Outs"
                >
                  {[1, 2].map((n) => (
                    <label
                      key={n}
                      className="outs-option"
                      title={`${n} out${n === 1 ? "" : "s"}`}
                    >
                      <input
                        type="checkbox"
                        name="outs"
                        checked={outs >= n}
                        onChange={() => {
                          // Two-light scoreboard behavior:
                          // none lit => 0 outs
                          // left lit => 1 out
                          // both lit => 2 outs
                          if (n === 1) {
                            setOuts(outs >= 1 ? 0 : 1);
                          } else {
                            setOuts(outs === 2 ? 1 : 2);
                          }
                        }}
                        aria-label={`${n} out${n === 1 ? "" : "s"}`}
                      />
                      <span className="outs-light" aria-hidden="true" />
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <div className="score">
              <p>Score:</p>
            </div>
            <div className="baserunners">
              <p>Baserunners:</p>
            </div>
            <div className="inning">
              <p>Inning:</p>
            </div>
          </div>
        </div>

        <button className="play-button" onClick={() => navigate("/simulator")}>
          Play Ball!
        </button>
      </div>
    </div>
  );
}
