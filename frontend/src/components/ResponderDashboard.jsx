import React, { useState, useEffect, useRef } from "react";
import { 
  Shield, Check, X, Navigation, MapPin, AlertTriangle, 
  Clock, HeartPulse, Flame, Activity, CheckCircle2,
  RefreshCw, Power, FastForward, Phone, AlertOctagon, LogOut,
  Volume2, VolumeX, Radio, Play, Pause, Compass
} from "lucide-react";
import MapComponent from "./MapComponent";
import { incidentApi, responderApi, routingApi, socket, deduplicateIncidents } from "../services/api";
import { sounds } from "../services/soundEffects";

export default function ResponderDashboard({ currentUser, onLogout }) {
  const [incidents, setIncidents] = useState([]);
  const [activeIncident, setActiveIncident] = useState(null);
  const [routeCoords, setRouteCoords] = useState([]);
  const [routeStats, setRouteStats] = useState({ distanceKm: 0, durationMinutes: 0 });
  const [isAvailable, setIsAvailable] = useState(true);

  // Auto-detected Responder GPS location
  const [responderCoords, setResponderCoords] = useState({ lat: 17.5950, lng: 78.4950 });
  const [isLocating, setIsLocating] = useState(false);
  const [isSimulatingMovement, setIsSimulatingMovement] = useState(false);
  const simIntervalRef = useRef(null);
  const geoWatchIdRef = useRef(null);

  // Incoming Emergency Alert Modal
  const [incomingAlert, setIncomingAlert] = useState(null);
  const [countdown, setCountdown] = useState(300); // 5 minutes

  const serviceType = currentUser?.service_type || "Ambulance";
  const responderName = currentUser?.full_name || `${serviceType} Officer`;

  useEffect(() => {
    loadIncidents();
    handleDetectGPS();

    if (currentUser?.responderId) {
      socket.emit("join_responder", currentUser.responderId);
    }
    socket.emit("join_dispatch");

    // 1. Direct 5-Minute Escalation incoming job alert
    socket.on("incoming_job_alert", (data) => {
      sounds.playAlertSiren();
      sounds.showSystemNotification("🚨 Incoming Emergency Dispatch!", `Incident ${data.incidentId} matches your ${serviceType} unit.`);
      setIncomingAlert(data);
      setCountdown(data.timeoutSeconds || 300);
    });

    // 2. Real-time Emergency SOS broadcast from any citizen
    socket.on("incident_created", (newInc) => {
      loadIncidents();
      sounds.playAlertSiren();
      
      // Match responder service or universal rescue
      const matchesService = 
        !newInc.suggested_service || 
        newInc.suggested_service === serviceType || 
        serviceType === "Rescue" || 
        (serviceType === "Police" && (newInc.emergency_type === "Crime" || newInc.emergency_type === "Police")) ||
        (serviceType === "Ambulance" && (newInc.emergency_type === "Medical" || newInc.emergency_type === "Crash")) ||
        (serviceType === "Fire" && (newInc.emergency_type === "Fire"));

      if (matchesService) {
        sounds.showSystemNotification("🚨 New Emergency SOS Reported!", `${newInc.emergency_type} incident reported at ${newInc.address || "GPS location"}`);
        setIncomingAlert({
          incidentId: newInc.id,
          incident: newInc,
          emergency_type: newInc.emergency_type,
          description: newInc.description,
          lat: newInc.lat,
          lng: newInc.lng,
          address: newInc.address,
          timeoutSeconds: 300
        });
        setCountdown(300);
      }
    });

    socket.on("job_offer_expired", () => {
      setIncomingAlert(null);
    });

    socket.on("incident_status_changed", () => {
      loadIncidents();
    });

    return () => {
      socket.off("incoming_job_alert");
      socket.off("incident_created");
      socket.off("job_offer_expired");
      socket.off("incident_status_changed");
      if (simIntervalRef.current) clearInterval(simIntervalRef.current);
      if (geoWatchIdRef.current && navigator.geolocation) {
        navigator.geolocation.clearWatch(geoWatchIdRef.current);
      }
    };
  }, [currentUser, serviceType]);

  // Countdown timer for incoming escalation alert
  useEffect(() => {
    let timer;
    if (incomingAlert && countdown > 0) {
      timer = setInterval(() => {
        setCountdown((c) => c - 1);
        if (countdown % 10 === 0) {
          sounds.playStep();
        }
      }, 1000);
    } else if (countdown === 0 && incomingAlert) {
      setIncomingAlert(null);
    }
    return () => clearInterval(timer);
  }, [incomingAlert, countdown]);

  const handleDetectGPS = () => {
    sounds.playTap();
    setIsLocating(true);
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = parseFloat(pos.coords.latitude.toFixed(5));
          const lng = parseFloat(pos.coords.longitude.toFixed(5));
          setResponderCoords({ lat, lng });
          setIsLocating(false);
          responderApi.updateLocation(lat, lng);
        },
        () => {
          setResponderCoords({ lat: 17.5950, lng: 78.4950 });
          setIsLocating(false);
        },
        { enableHighAccuracy: true, timeout: 5000 }
      );
    } else {
      setIsLocating(false);
    }
  };

  const startLiveGpsBroadcasting = (incidentId) => {
    if (navigator.geolocation) {
      if (geoWatchIdRef.current) navigator.geolocation.clearWatch(geoWatchIdRef.current);
      geoWatchIdRef.current = navigator.geolocation.watchPosition(
        (pos) => {
          const lat = parseFloat(pos.coords.latitude.toFixed(5));
          const lng = parseFloat(pos.coords.longitude.toFixed(5));
          setResponderCoords({ lat, lng });
          socket.emit("live_gps_stream", {
            incidentId,
            lat,
            lng,
            heading: pos.coords.heading || 0,
            responderName
          });
        },
        (err) => console.warn("GPS watch error", err),
        { enableHighAccuracy: true, maximumAge: 2000 }
      );
    }
  };

  const loadIncidents = async () => {
    try {
      const data = await incidentApi.list();
      const clean = deduplicateIncidents(data || []);
      setIncidents(clean);

      const assigned = clean.find(i => (i.assigned_responder_id === currentUser?.responderId || i.assigned_responder?.id === currentUser?.responderId) && i.status !== "Resolved");
      if (assigned) {
        setActiveIncident(assigned);
        fetchRoute(assigned);
        startLiveGpsBroadcasting(assigned.id);
      }
    } catch (e) {
      console.warn("Failed to load responder incidents", e);
    }
  };

  const fetchRoute = async (incident) => {
    try {
      const res = await routingApi.getRoute(responderCoords.lat, responderCoords.lng, incident.lat, incident.lng);
      if (res && res.coordinates) {
        setRouteCoords(res.coordinates);
        setRouteStats({
          distanceKm: res.distance_km || 2.1,
          durationMinutes: res.duration_minutes || 5
        });
      }
    } catch (e) {
      console.warn(e);
    }
  };

  const handleAccept = async (incidentId) => {
    sounds.playSuccess();
    try {
      const res = await incidentApi.assign(incidentId, "accept", responderCoords.lat, responderCoords.lng);
      setIncomingAlert(null);
      const inc = res.incident || incidents.find(i => i.id === incidentId);
      if (inc) {
        if (inc.assigned_responder) {
          inc.assigned_responder.lat = responderCoords.lat;
          inc.assigned_responder.lng = responderCoords.lng;
        }
        setActiveIncident(inc);
        fetchRoute(inc);
        startLiveGpsBroadcasting(inc.id);
      }
      
      // Immediately broadcast responder GPS start location
      socket.emit("live_gps_stream", {
        incidentId,
        lat: responderCoords.lat,
        lng: responderCoords.lng,
        responderName
      });

      loadIncidents();
    } catch (err) {
      alert(err.response?.data?.error || "Incident already accepted or unavailable.");
      setIncomingAlert(null);
      loadIncidents();
    }
  };

  const handleDecline = async (incidentId) => {
    sounds.playTap();
    try {
      await incidentApi.assign(incidentId, "decline");
      setIncomingAlert(null);
      loadIncidents();
    } catch (err) {
      setIncomingAlert(null);
    }
  };

  const handleStatusChange = async (nextStatus) => {
    if (!activeIncident) return;
    if (nextStatus === "Resolved") {
      sounds.playSuccess();
      if (simIntervalRef.current) clearInterval(simIntervalRef.current);
      setIsSimulatingMovement(false);
    } else {
      sounds.playStep();
    }
    try {
      const res = await incidentApi.updateStatus(
        activeIncident.id,
        nextStatus,
        `Status updated to ${nextStatus}`,
        responderCoords.lat,
        responderCoords.lng,
        nextStatus === "Resolved" ? "Handled & Stabilized" : null
      );
      setActiveIncident(res.incident || { ...activeIncident, status: nextStatus });
      loadIncidents();
    } catch (err) {
      setActiveIncident({ ...activeIncident, status: nextStatus });
    }
  };

  // Real-time simulated movement towards citizen
  const simulateLiveMovement = () => {
    if (!activeIncident || !routeCoords || routeCoords.length < 2) return;
    sounds.playStep();
    setIsSimulatingMovement(true);

    if (activeIncident.status === "Reported" || activeIncident.status === "Assigned") {
      handleStatusChange("En Route");
    }

    let step = 0;
    if (simIntervalRef.current) clearInterval(simIntervalRef.current);
    simIntervalRef.current = setInterval(() => {
      if (step < routeCoords.length) {
        const [lat, lng] = routeCoords[step];
        setResponderCoords({ lat, lng });
        socket.emit("live_gps_stream", {
          incidentId: activeIncident.id,
          lat,
          lng,
          responderName
        });
        step += 1;
      } else {
        clearInterval(simIntervalRef.current);
        setIsSimulatingMovement(false);
        handleStatusChange("On Scene");
      }
    }, 1200);
  };

  const stopSimulatedMovement = () => {
    if (simIntervalRef.current) clearInterval(simIntervalRef.current);
    setIsSimulatingMovement(false);
  };

  const unassignedMatchingPool = incidents.filter(i => 
    (i.status === "Reported" || i.status === "Awaiting Responder") && 
    (i.suggested_service === serviceType || serviceType === "Rescue" || !i.suggested_service)
  );

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "16px", display: "flex", flexDirection: "column", gap: "16px" }}>
      
      {/* 5-Minute Flashing Incoming Job Alert Modal */}
      {incomingAlert && (
        <div className="modal-overlay">
          <div className="modal-container" style={{
            maxWidth: "490px",
            padding: "26px",
            textAlign: "center",
            border: "2px solid #ff334b",
            background: "#0e1424",
            boxShadow: "0 0 45px rgba(255, 51, 75, 0.6)",
            animation: "pulse 1.2s infinite"
          }}>
            <div style={{ width: "64px", height: "64px", borderRadius: "50%", background: "rgba(255, 51, 75, 0.25)", color: "#ff334b", display: "inline-flex", alignItems: "center", justifyContent: "center", marginBottom: "12px" }}>
              <AlertOctagon size={32} className="animate-spin" />
            </div>

            <div style={{ fontSize: "2.2rem", fontWeight: "900", color: "#ff334b", fontFamily: "monospace", marginBottom: "2px" }}>
              {Math.floor(countdown / 60).toString().padStart(2, '0')}:{(countdown % 60).toString().padStart(2, '0')}
            </div>
            <div style={{ fontSize: "0.75rem", color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "14px", fontWeight: "700" }}>
              🚨 Real-Time Dispatch Alert • 5-Minute Acceptance Timer
            </div>

            <h3 style={{ fontSize: "1.25rem", fontWeight: "900", color: "#f8fafc", marginBottom: "6px" }}>
              {incomingAlert.emergency_type || "Emergency"} Incident Reported!
            </h3>
            
            <div style={{ background: "rgba(255,255,255,0.05)", padding: "12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.1)", marginBottom: "18px", textAlign: "left", fontSize: "0.82rem", color: "#cbd5e1" }}>
              <div><strong>Ticket ID:</strong> {incomingAlert.incidentId}</div>
              {incomingAlert.description && <div style={{ marginTop: "4px" }}><strong>Details:</strong> {incomingAlert.description}</div>}
              {incomingAlert.address && <div style={{ marginTop: "4px" }}><strong>Location:</strong> {incomingAlert.address}</div>}
            </div>

            <div style={{ display: "flex", gap: "10px" }}>
              <button
                onClick={() => handleAccept(incomingAlert.incidentId)}
                className="btn-emergency-main"
                style={{ flex: 1, padding: "12px", fontSize: "0.95rem", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
              >
                <Check size={18} /> Accept Emergency
              </button>
              <button
                onClick={() => handleDecline(incomingAlert.incidentId)}
                className="btn-outline"
                style={{ padding: "12px 18px" }}
              >
                Decline
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Top Console Bar */}
      <header style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: "12px",
        padding: "14px 18px",
        background: "rgba(14, 20, 36, 0.95)",
        borderRadius: "14px",
        border: "1px solid rgba(255,255,255,0.1)",
        boxShadow: "0 4px 20px rgba(0,0,0,0.4)"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{ width: "36px", height: "36px", borderRadius: "10px", background: "rgba(0, 229, 255, 0.18)", color: "#00e5ff", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Radio size={20} className="animate-pulse" />
          </div>
          <div>
            <div style={{ fontSize: "1.1rem", fontWeight: "900", color: "#f8fafc" }}>
              {serviceType} Responder Unit
            </div>
            <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>
              {currentUser?.full_name || "Official Unit"} • Vehicle: {currentUser?.vehicle_number || "DEMO-UNIT"}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <button
            onClick={handleDetectGPS}
            className="btn-outline"
            style={{ padding: "8px 12px", fontSize: "0.78rem", display: "flex", alignItems: "center", gap: "5px" }}
            title="Calibrate GPS"
          >
            <Compass size={14} className={isLocating ? "animate-spin" : ""} />
            <span>GPS: {responderCoords.lat.toFixed(3)}, {responderCoords.lng.toFixed(3)}</span>
          </button>

          <button
            onClick={() => { sounds.playTap(); onLogout(); }}
            className="btn-outline"
            style={{ padding: "8px 12px", fontSize: "0.78rem" }}
            title="Log out"
          >
            <LogOut size={14} />
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "16px" }}>
        
        {/* Left Column: Active Job & Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          
          {activeIncident ? (
            <div style={{
              background: "rgba(14, 20, 36, 0.95)",
              border: "1.5px solid rgba(0, 229, 255, 0.35)",
              borderRadius: "16px",
              padding: "18px",
              display: "flex",
              flexDirection: "column",
              gap: "14px",
              boxShadow: "0 8px 30px rgba(0,0,0,0.5)"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <span className="neon-badge neon-badge-critical" style={{ fontSize: "0.72rem" }}>
                    ● ACTIVE EMERGENCY MISSION
                  </span>
                  <h3 style={{ fontSize: "1.2rem", fontWeight: "900", color: "#f8fafc", marginTop: "4px" }}>
                    {activeIncident.id} • {activeIncident.emergency_type}
                  </h3>
                </div>
                <span className="neon-badge neon-badge-enroute" style={{ fontSize: "0.8rem", padding: "4px 10px" }}>
                  {activeIncident.status}
                </span>
              </div>

              {activeIncident.description && (
                <div style={{ background: "rgba(255,255,255,0.04)", padding: "10px 12px", borderRadius: "8px", fontSize: "0.82rem", color: "#e2e8f0" }}>
                  <strong>Description: </strong>{activeIncident.description}
                </div>
              )}

              {activeIncident.address && (
                <div style={{ fontSize: "0.78rem", color: "#94a3b8", display: "flex", alignItems: "center", gap: "6px" }}>
                  <MapPin size={14} color="#00e5ff" />
                  <span>{activeIncident.address}</span>
                </div>
              )}

              {/* Status Stepper Actions */}
              <div style={{ display: "flex", flexDirection: "column", gap: "8px", paddingTop: "6px", borderTop: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.75rem", fontWeight: "800", color: "#94a3b8", textTransform: "uppercase" }}>
                  Mission Actions:
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                  {activeIncident.status !== "En Route" && activeIncident.status !== "On Scene" && (
                    <button
                      onClick={() => handleStatusChange("En Route")}
                      className="btn-outline"
                      style={{ padding: "10px", fontSize: "0.82rem", borderColor: "#00e5ff", color: "#00e5ff", fontWeight: "800" }}
                    >
                      Mark En Route
                    </button>
                  )}

                  {activeIncident.status !== "On Scene" && (
                    <button
                      onClick={() => handleStatusChange("On Scene")}
                      className="btn-outline"
                      style={{ padding: "10px", fontSize: "0.82rem", borderColor: "#eab308", color: "#eab308", fontWeight: "800" }}
                    >
                      Arrived On Scene
                    </button>
                  )}

                  <button
                    onClick={() => handleStatusChange("Resolved")}
                    style={{
                      gridColumn: "span 2",
                      background: "linear-gradient(135deg, #00ff88, #059669)",
                      color: "#070a12",
                      border: "none",
                      borderRadius: "8px",
                      padding: "10px",
                      fontSize: "0.88rem",
                      fontWeight: "900",
                      cursor: "pointer"
                    }}
                  >
                    <CheckCircle2 size={16} style={{ display: "inline", verticalAlign: "middle", marginRight: "4px" }} />
                    Complete & Resolve Emergency
                  </button>
                </div>

                {/* Real-Time Driving Simulation Button */}
                <button
                  onClick={isSimulatingMovement ? stopSimulatedMovement : simulateLiveMovement}
                  style={{
                    marginTop: "6px",
                    background: isSimulatingMovement ? "rgba(255, 51, 75, 0.2)" : "rgba(0, 229, 255, 0.15)",
                    border: isSimulatingMovement ? "1px solid #ff334b" : "1px solid #00e5ff",
                    color: isSimulatingMovement ? "#ff4d67" : "#00e5ff",
                    borderRadius: "8px",
                    padding: "10px",
                    fontSize: "0.82rem",
                    fontWeight: "800",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "6px"
                  }}
                >
                  {isSimulatingMovement ? (
                    <>
                      <Pause size={15} /> Pause Real-Time Movement
                    </>
                  ) : (
                    <>
                      <Play size={15} /> 🚀 Drive Towards Citizen (Stream Live GPS)
                    </>
                  )}
                </button>
                
                {isSimulatingMovement && (
                  <div style={{ fontSize: "0.72rem", color: "#00ff88", textAlign: "center", animation: "pulse 1s infinite" }}>
                    ● Streaming live GPS coordinates to citizen in real-time...
                  </div>
                )}
              </div>

            </div>
          ) : (
            <div style={{
              background: "rgba(14, 20, 36, 0.95)",
              borderRadius: "16px",
              padding: "30px 20px",
              textAlign: "center",
              border: "1px solid rgba(255,255,255,0.08)"
            }}>
              <CheckCircle2 size={36} color="#00ff88" style={{ margin: "0 auto 10px auto" }} />
              <h3 style={{ fontSize: "1.1rem", fontWeight: "800", color: "#f8fafc" }}>
                Unit On Standby
              </h3>
              <p style={{ fontSize: "0.78rem", color: "#94a3b8", maxWidth: "280px", margin: "4px auto 0 auto" }}>
                Listening for emergency dispatches matching {serviceType} response.
              </p>
            </div>
          )}

          {/* Incoming Dispatch Queue */}
          {unassignedMatchingPool.length > 0 && !activeIncident && (
            <div style={{
              background: "rgba(14, 20, 36, 0.95)",
              borderRadius: "16px",
              padding: "16px",
              border: "1px solid rgba(255, 184, 0, 0.3)",
              display: "flex",
              flexDirection: "column",
              gap: "10px"
            }}>
              <div style={{ fontSize: "0.85rem", fontWeight: "800", color: "#ffb800", display: "flex", alignItems: "center", gap: "6px" }}>
                <AlertTriangle size={16} /> Incoming Emergency Dispatches ({unassignedMatchingPool.length})
              </div>

              {unassignedMatchingPool.map((inc) => (
                <div key={inc.id} style={{ background: "rgba(30, 41, 59, 0.6)", padding: "12px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.08)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ fontWeight: "800", color: "#f8fafc", fontSize: "0.85rem" }}>{inc.id} • {inc.emergency_type}</div>
                    <div style={{ fontSize: "0.72rem", color: "#94a3b8" }}>{inc.address || "GPS Coordinates"}</div>
                  </div>
                  <button
                    onClick={() => handleAccept(inc.id)}
                    className="btn-emergency-main"
                    style={{ padding: "6px 12px", fontSize: "0.78rem" }}
                  >
                    Accept
                  </button>
                </div>
              ))}
            </div>
          )}

        </div>

        {/* Right Column: Live Map */}
        <div style={{
          height: "70vh",
          minHeight: "420px",
          borderRadius: "16px",
          overflow: "hidden",
          border: "1.5px solid rgba(0, 229, 255, 0.3)",
          boxShadow: "0 8px 30px rgba(0,0,0,0.5)",
          position: "relative"
        }}>
          <MapComponent
            height="100%"
            center={[responderCoords.lat, responderCoords.lng]}
            zoom={14}
            incidentLocation={activeIncident ? { lat: activeIncident.lat, lng: activeIncident.lng } : null}
            incidentLabel={activeIncident ? `${activeIncident.id} (${activeIncident.emergency_type})` : ""}
            responderLocation={{ lat: responderCoords.lat, lng: responderCoords.lng }}
            responderType={serviceType}
            responderLabel={`${responderName} (Live GPS)`}
            routeCoordinates={routeCoords}
          />
        </div>

      </div>

    </div>
  );
}
