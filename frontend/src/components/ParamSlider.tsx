interface Props {
  label: string;
  param: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

export default function ParamSlider({ label, value, min, max, step, onChange }: Props) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-[10px] text-text-secondary w-20 shrink-0">{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-0.5 accent-accent"
      />
      <span className="text-[10px] font-mono text-text-secondary w-10 text-right">{value}</span>
    </div>
  );
}
