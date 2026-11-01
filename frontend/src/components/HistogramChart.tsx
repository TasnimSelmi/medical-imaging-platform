import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer } from "recharts";

interface Props {
  data: Record<string, number[]>;
}

const CHANNEL_COLORS: Record<string, string> = {
  gray: "#6b6b6b",
  r: "#dc2626",
  g: "#16a34a",
  b: "#2563eb",
};

export default function HistogramChart({ data }: Props) {
  const channels = Object.keys(data);
  const chartData = Array.from({ length: 256 }, (_, i) => {
    const point: Record<string, number> = { bin: i };
    for (const ch of channels) {
      point[ch] = data[ch][i];
    }
    return point;
  });

  return (
    <div className="mt-2 border border-border rounded p-2 bg-bg">
      <div className="text-[10px] text-text-secondary mb-1">Histogramme</div>
      <ResponsiveContainer width="100%" height={80}>
        <AreaChart data={chartData}>
          <XAxis dataKey="bin" hide />
          <YAxis hide />
          {channels.map((ch) => (
            <Area
              key={ch}
              type="monotone"
              dataKey={ch}
              stroke={CHANNEL_COLORS[ch] || "#6b6b6b"}
              fill={CHANNEL_COLORS[ch] || "#6b6b6b"}
              fillOpacity={0.15}
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
