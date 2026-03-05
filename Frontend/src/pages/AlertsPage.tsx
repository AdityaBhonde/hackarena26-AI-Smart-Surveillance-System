import React, { useEffect, useState, useMemo } from "react";

type Clip = {
  name: string;
  url: string;
  type: string;
  timestamp: string;
  date: string;
  rawTime: string;
};

const AlertsPage: React.FC = () => {
  const [clips, setClips] = useState<Clip[]>([]);
  const [filter, setFilter] = useState("All");
  const [searchDate, setSearchDate] = useState("");
  const [searchTime, setSearchTime] = useState("");

  const fetchClips = async () => {
    try {
      const res = await fetch("http://localhost:5000/clips");
      const data = await res.json();
      
      const parsedData = data.map((clip: any) => {
        // ✅ FIX: Split using '--' to match your stable Python recorder logic
        const parts = clip.name.split('--'); 
        
        // parts[0] is alert_type (e.g., "Weapon")
        // parts[1] is the date (e.g., "2026-03-05")
        // parts[2] is the time (e.g., "23-46-51.mp4")
        const alertType = parts[0] || "Unknown";
        const datePart = parts[1] || "";
        const timePart = parts[2]?.replace('.mp4', '') || "";

        return {
          ...clip,
          type: alertType,
          date: datePart, 
          rawTime: timePart, 
          // Formats the time from 23-46-51 to 23:46:51 for the UI
          timestamp: timePart.replace(/-/g, ':')
        };
      });
      setClips(parsedData);
    } catch (err) {
      console.error("Error loading clips", err);
    }
  };

  useEffect(() => {
    fetchClips();
    const interval = setInterval(fetchClips, 10000);
    return () => clearInterval(interval);
  }, []);

  const filteredClips = useMemo(() => {
    return clips.filter(clip => {
      const matchesType = filter === "All" || clip.type === filter;
      const matchesDate = !searchDate || clip.date === searchDate;
      const matchesTime = !searchTime || clip.timestamp.startsWith(searchTime);
      return matchesType && matchesDate && matchesTime;
    });
  }, [clips, filter, searchDate, searchTime]);

  const stats = {
    total: clips.length,
    weapons: clips.filter(c => c.type === "Weapon").length,
    crowds: clips.filter(c => c.type === "Crowd").length,
    loitering: clips.filter(c => c.type === "Loitering").length,
    criminal: clips.filter(c => c.type === "Criminal").length,
  };

  const getBadgeColor = (type: string) => {
    const colors: Record<string, string> = {
      Weapon: "#ef4444",   // Red
      Crowd: "#f59e0b",    // Orange
      Loitering: "#3b82f6", // Blue
      Criminal: "#a855f7"  // Purple
    };
    return colors[type] || "#64748b";
  };

  return (
    <div style={{ display: "flex", background: "#020617", minHeight: "100vh", color: "white", fontFamily: "'Inter', sans-serif" }}>
      
      {/* SIDEBAR */}
      <div style={{ width: "280px", background: "#0f172a", padding: "30px", borderRight: "1px solid #1e293b", position: "sticky", top: 0, height: "100vh" }}>
        <h2 style={{ fontSize: "22px", fontWeight: "800", marginBottom: "30px", color: "#38bdf8" }}>🛡️ MONITOR</h2>
        
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div>
            <label style={{ fontSize: "11px", color: "#64748b", fontWeight: "bold", letterSpacing: "1px" }}>EVENT DATE</label>
            <input 
              type="date" 
              value={searchDate}
              onChange={(e) => setSearchDate(e.target.value)}
              style={{ width: "100%", background: "#1e293b", border: "1px solid #334155", color: "white", padding: "12px", borderRadius: "10px", marginTop: "8px" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "11px", color: "#64748b", fontWeight: "bold", letterSpacing: "1px" }}>EVENT TIME (HH:MM)</label>
            <input 
              type="time" 
              value={searchTime}
              onChange={(e) => setSearchTime(e.target.value)}
              style={{ width: "100%", background: "#1e293b", border: "1px solid #334155", color: "white", padding: "12px", borderRadius: "10px", marginTop: "8px" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "11px", color: "#64748b", fontWeight: "bold", letterSpacing: "1px" }}>THREAT TYPE</label>
            {["All", "Weapon", "Crowd", "Loitering", "Criminal"].map(t => (
              <button 
                key={t}
                onClick={() => setFilter(t)}
                style={{ 
                  display: "block", width: "100%", textAlign: "left", padding: "12px", marginTop: "8px", 
                  borderRadius: "10px", border: "none", cursor: "pointer", transition: "0.2s",
                  background: filter === t ? getBadgeColor(t === "All" ? "" : t) : "#1e293b",
                  color: filter === t ? "white" : "#94a3b8",
                  fontSize: "14px"
                }}
              >
                {t === "All" ? "📜 All Logs" : t === "Criminal" ? "👤 Wanted Person" : t}
              </button>
            ))}
          </div>
          
          <button 
            onClick={() => {setSearchDate(""); setSearchTime(""); setFilter("All")}}
            style={{ marginTop: "20px", background: "transparent", border: "1px dashed #334155", color: "#64748b", padding: "10px", borderRadius: "10px", cursor: "pointer" }}
          >
            Reset All Filters
          </button>
        </div>
      </div>

      {/* MAIN PANEL */}
      <div style={{ flex: 1, padding: "40px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "15px", marginBottom: "40px" }}>
          {[
            { label: "Total Alerts", val: stats.total, col: "#fff" },
            { label: "Weapons", val: stats.weapons, col: "#ef4444" },
            { label: "Criminals", val: stats.criminal, col: "#a855f7" },
            { label: "Crowds", val: stats.crowds, col: "#f59e0b" },
            { label: "Loitering", val: stats.loitering, col: "#3b82f6" }
          ].map((s, i) => (
            <div key={i} style={{ background: "#0f172a", padding: "20px", borderRadius: "16px", border: "1px solid #1e293b" }}>
              <p style={{ margin: 0, color: "#64748b", fontSize: "10px", fontWeight: "bold" }}>{s.label.toUpperCase()}</p>
              <h3 style={{ margin: "5px 0 0 0", fontSize: "24px", color: s.col }}>{s.val}</h3>
            </div>
          ))}
        </div>

        <header style={{ marginBottom: "30px" }}>
          <h1 style={{ fontSize: "32px", fontWeight: "800", letterSpacing: "-1px" }}>Incident Archive</h1>
          <p style={{ color: "#64748b" }}>Advanced detection log for Aditya's Security System</p>
        </header>

        {filteredClips.length === 0 ? (
          <div style={{ padding: "100px", textAlign: "center", background: "#0f172a", borderRadius: "20px", border: "1px dashed #334155" }}>
            <p style={{ fontSize: "40px" }}>📡</p>
            <p style={{ color: "#64748b" }}>No incidents found matching these parameters.</p>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(420px, 1fr))", gap: "25px" }}>
            {filteredClips.map((clip, index) => (
              <div key={index} style={{ 
                background: "#0f172a", 
                borderRadius: "20px", 
                overflow: "hidden", 
                border: clip.type === "Criminal" ? "1px solid #a855f7" : "1px solid #1e293b",
                boxShadow: clip.type === "Criminal" ? "0 0 15px rgba(168, 85, 247, 0.2)" : "0 10px 15px -3px rgba(0, 0, 0, 0.1)" 
              }}>
                <div style={{ padding: "20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{ 
                      width: "10px", 
                      height: "10px", 
                      borderRadius: "50%", 
                      background: getBadgeColor(clip.type),
                      boxShadow: `0 0 8px ${getBadgeColor(clip.type)}` 
                    }}></div>
                    <span style={{ fontWeight: "bold", fontSize: "14px" }}>{clip.type === "Criminal" ? "WANTED PERSON" : clip.type}</span>
                  </div>
                  <span style={{ fontSize: "12px", color: "#64748b", fontFamily: "monospace" }}>{clip.date} | {clip.timestamp}</span>
                </div>

                <video controls muted width="100%" style={{ background: "black", aspectRatio: "16/9" }}>
                  <source src={clip.url} type="video/mp4" />
                </video>

                <div style={{ padding: "20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "11px", color: "#334155" }}>REF: {clip.name}</span>
                  <a href={clip.url} download style={{ background: getBadgeColor(clip.type), color: "white", textDecoration: "none", padding: "8px 16px", borderRadius: "8px", fontSize: "12px", fontWeight: "bold" }}>
                    Export Evidence
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default AlertsPage;
