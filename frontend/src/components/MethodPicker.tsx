interface Method {
  key: string;
  label: string;
}

interface Props {
  methods: Method[];
  onSelect: (key: string) => void;
  loading?: boolean;
}

export default function MethodPicker({ methods, onSelect, loading }: Props) {
  return (
    <div className="flex flex-col gap-1">
      {methods.map((m) => (
        <button
          key={m.key}
          onClick={() => onSelect(m.key)}
          disabled={loading}
          className="text-left px-3 py-1.5 text-xs rounded border border-border bg-surface hover:bg-bg transition-colors duration-150 disabled:opacity-50"
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
