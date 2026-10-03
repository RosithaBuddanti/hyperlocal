import React, { useState, useEffect } from "react";
import LandingPage from "./components/LandingPage";
import AuthModal from "./components/AuthModal";
import SosModal from "./components/SosModal";
import CitizenDashboard from "./components/CitizenDashboard";
import ResponderDashboard from "./components/ResponderDashboard";
import { authApi, removeDuplicateData } from "./services/api";

export default function App() {
  // Navigation views: 'landing' | 'citizen-dashboard' | 'responder-dashboard'
  const [currentView, setCurrentView] = useState("landing");
  
  // Auth state
  const [currentUser, setCurrentUser] = useState(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState("citizen-login");

  // SOS Modal state
  const [isSosOpen, setIsSosOpen] = useState(false);
  const [submittedIncident, setSubmittedIncident] = useState(null);

  useEffect(() => {
    // 1. Proactively purge any duplicate entries across localStorage on mount
    removeDuplicateData();

    // 2. Check if previously logged in user token exists
    const storedUser = localStorage.getItem("aegis_user") || localStorage.getItem("emergency_user");
    if (storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setCurrentUser(user);
        if (user.role === "citizen") setCurrentView("citizen-dashboard");
        else if (user.role === "responder") setCurrentView("responder-dashboard");
      } catch (e) {
        localStorage.removeItem("aegis_user");
        localStorage.removeItem("emergency_user");
      }
    }
  }, []);

  const handleOpenAuth = (mode) => {
    setAuthMode(mode);
    setIsAuthOpen(true);
  };

  const handleAuthSuccess = (user) => {
    setCurrentUser(user);
    localStorage.setItem("aegis_user", JSON.stringify(user));
    if (user.role === "citizen") {
      setCurrentView("citizen-dashboard");
    } else if (user.role === "responder") {
      setCurrentView("responder-dashboard");
    }
  };

  const handleContinueAsGuest = () => {
    const guestUser = {
      id: "guest_" + Date.now().toString(36),
      full_name: "Guest Citizen",
      role: "citizen",
      isGuest: true
    };
    setCurrentUser(guestUser);
    setCurrentView("citizen-dashboard");
  };

  const handleLogout = () => {
    authApi.logout();
    localStorage.removeItem("aegis_user");
    setCurrentUser(null);
    setSubmittedIncident(null);
    setCurrentView("landing");
  };

  const handleSosSubmitted = (incident) => {
    if (!currentUser) {
      const guestUser = {
        id: "guest_" + Date.now().toString(36),
        full_name: "Guest Citizen",
        role: "citizen",
        isGuest: true
      };
      setCurrentUser(guestUser);
    }
    setSubmittedIncident(incident);
    setCurrentView("citizen-dashboard");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#070a12", color: "#f8fafc" }}>
      
      {/* 1. Landing Page (Requirement 1) */}
      {currentView === "landing" && (
        <LandingPage
          onContinueAsGuest={handleContinueAsGuest}
          onOpenCitizenLogin={() => handleOpenAuth("citizen-login")}
          onOpenCitizenRegister={() => handleOpenAuth("citizen-register")}
          onOpenResponderLogin={() => handleOpenAuth("responder-login")}
          onOpenResponderRegister={() => handleOpenAuth("responder-register")}
          onTriggerSos={() => setIsSosOpen(true)}
        />
      )}

      {/* 2. Citizen Dashboard (Requirements 6 & 7) */}
      {currentView === "citizen-dashboard" && (
        <CitizenDashboard
          currentUser={currentUser}
          initialIncident={submittedIncident}
          onOpenSos={() => setIsSosOpen(true)}
          onLogout={handleLogout}
        />
      )}

      {/* 3. Responder Dashboard (Requirements 5, 6, & 7) */}
      {currentView === "responder-dashboard" && (
        <ResponderDashboard
          currentUser={currentUser}
          onLogout={handleLogout}
        />
      )}

      {/* Global Auth Modal */}
      <AuthModal
        isOpen={isAuthOpen}
        initialMode={authMode}
        onClose={() => setIsAuthOpen(false)}
        onAuthSuccess={handleAuthSuccess}
      />

      {/* Global SOS Emergency Form Modal (Requirement 2 & 3) */}
      <SosModal
        isOpen={isSosOpen}
        onClose={() => setIsSosOpen(false)}
        onSubmitted={handleSosSubmitted}
      />

    </div>
  );
}
