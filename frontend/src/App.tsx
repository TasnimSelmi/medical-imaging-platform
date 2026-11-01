import { Routes, Route, useLocation, Link } from "react-router-dom";
import Home from "./pages/Home";
import Workspace from "./pages/Workspace";
import Compare from "./pages/Compare";
import Uncertainty from "./pages/Uncertainty";

const NAV: { path: string; label: string }[] = [
  { path: "/workspace", label: "Espace de travail" },
  { path: "/compare", label: "Comparaison" },
  { path: "/uncertainty", label: "Incertitude" },
];

export default function App() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-bg">
      <header className="h-12 border-b border-border bg-surface flex items-center px-4 justify-between">
        <div className="flex items-center gap-6">
          <Link to="/" className="text-sm font-semibold tracking-tight text-text">
            Medical Imaging Platform
          </Link>
          <nav className="flex items-center gap-1">
            {NAV.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-2 py-1 text-xs rounded transition-colors duration-150 ${
                  location.pathname === item.path
                    ? "text-accent bg-blue-50"
                    : "text-text-secondary hover:text-text hover:bg-bg"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
        <span className="text-xs text-text-secondary font-mono">session</span>
      </header>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/workspace" element={<Workspace />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/uncertainty" element={<Uncertainty />} />
      </Routes>
    </div>
  );
}
