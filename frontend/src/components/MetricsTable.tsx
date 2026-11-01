interface Metrics {
  iou?: number;
  dice?: number;
  pearson?: number;
  mse?: number;
  nmi?: number;
}

interface Props {
  metrics: Metrics | null;
}

export default function MetricsTable({ metrics }: Props) {
  if (!metrics) return null;

  const rows = [
    { label: "IoU", value: metrics.iou },
    { label: "Dice", value: metrics.dice },
    { label: "Pearson", value: metrics.pearson },
    { label: "MSE", value: metrics.mse },
    { label: "NMI", value: metrics.nmi },
  ].filter((r) => r.value !== undefined);

  return (
    <table className="w-full text-xs">
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} className="border-b border-border last:border-0">
            <td className="py-1 text-text-secondary">{r.label}</td>
            <td className="py-1 text-right font-mono">{r.value!.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
