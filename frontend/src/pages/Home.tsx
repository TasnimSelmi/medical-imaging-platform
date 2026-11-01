import { useNavigate } from "react-router-dom";
import { Brain, Stethoscope } from "lucide-react";

export default function Home() {
  const navigate = useNavigate();

  return (
    <main className="flex flex-col items-center justify-center px-4" style={{ minHeight: "calc(100vh - 48px)" }}>
      <h1 className="text-2xl font-semibold tracking-tight text-text mb-2">
        Plateforme d'analyse d'images médicales
      </h1>
      <p className="text-sm text-text-secondary mb-10">
        Prétraitement, segmentation classique, apprentissage profond et quantification d'incertitude
      </p>

      <div className="flex gap-4">
        <button
          onClick={() => {
            navigate("/workspace?modality=brain");
          }}
          className="w-56 border border-border rounded-md p-6 bg-surface hover:bg-bg transition-colors duration-150 text-left"
        >
          <Brain size={24} strokeWidth={1.5} className="text-accent mb-3" />
          <div className="text-sm font-medium text-text">IRM cérébrales</div>
          <div className="text-xs text-text-secondary mt-1">Prétraitement et segmentation conventionnelle</div>
        </button>

        <button
          onClick={() => {
            navigate("/workspace?modality=endoscopy");
          }}
          className="w-56 border border-border rounded-md p-6 bg-surface hover:bg-bg transition-colors duration-150 text-left"
        >
          <Stethoscope size={24} strokeWidth={1.5} className="text-accent mb-3" />
          <div className="text-sm font-medium text-text">Endoscopie</div>
          <div className="text-xs text-text-secondary mt-1">Détection de polypes</div>
        </button>
      </div>

      <a href="#about" className="text-xs text-text-secondary mt-12 hover:text-accent transition-colors duration-150">
        À propos
      </a>
    </main>
  );
}
