import { useEffect, useMemo, useState } from "react";
import Select from "react-select";
import { getPlayers, predict } from "../api";

export default function App() {
  const [players, setPlayers] = useState({ pitchers: [], batters: [] });
  const [pitcher, setPitcher] = useState(null); // store selected option object
  const [batter, setBatter] = useState(null);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await getPlayers();
        setPlayers(data);

        // default selections
        if (data.pitchers?.length)
          setPitcher({
            value: data.pitchers[0].id,
            label: data.pitchers[0].label,
          });
        if (data.batters?.length)
          setBatter({
            value: data.batters[0].id,
            label: data.batters[0].label,
          });
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

  async function onPlay() {
    setErr("");
    setResult(null);
    try {
      if (!pitcher || !batter)
        throw new Error("Select a pitcher and batter first.");
      const data = await predict(Number(pitcher.value), Number(batter.value));
      setResult(data);
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <div
      style={{ maxWidth: 800, margin: "40px auto", fontFamily: "system-ui" }}
    >
      <h1>At Bat Simulator</h1>

      <div style={{ display: "flex", gap: 16, marginTop: 20 }}>
        <div style={{ flex: 1 }}>
          <label>Pitcher</label>
          <div style={{ marginTop: 6 }}>
            <Select
              value={pitcher}
              onChange={setPitcher}
              options={pitcherOptions}
              placeholder="Search pitcher..."
              isClearable
              isSearchable
              styles={{
                singleValue: (base) => ({ ...base, color: "black" }),
                input: (base) => ({ ...base, color: "black" }),
                option: (base, state) => ({
                  ...base,
                  color: "black",
                  backgroundColor: state.isFocused ? "#eee" : "white",
                }),
              }}
            />
          </div>
        </div>

        <div style={{ flex: 1 }}>
          <label>Batter</label>
          <div style={{ marginTop: 6 }}>
            <Select
              value={batter}
              onChange={setBatter}
              options={batterOptions}
              placeholder="Search batter..."
              isClearable
              isSearchable
              styles={{
                singleValue: (base) => ({ ...base, color: "black" }),
                input: (base) => ({ ...base, color: "black" }),
                option: (base, state) => ({
                  ...base,
                  color: "black",
                  backgroundColor: state.isFocused ? "#eee" : "white",
                }),
              }}
            />
          </div>
        </div>
      </div>

      <button
        onClick={onPlay}
        style={{ marginTop: 16, padding: "10px 14px", cursor: "pointer" }}
      >
        Play
      </button>

      {err && (
        <div style={{ marginTop: 16, color: "crimson" }}>
          <b>Error:</b> {err}
        </div>
      )}

      {result && (
        <pre
          style={{
            marginTop: 16,
            padding: 12,
            background: "#f6f6f6",
            color: "black",
            borderRadius: 8,
            overflowX: "auto",
          }}
        >
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
