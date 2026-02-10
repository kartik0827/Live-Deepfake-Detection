import { useMemo } from 'react'
import {
    XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart,
} from 'recharts'

interface MetricsProps {
    fusionHistory: { time: number; score: number }[];
    sensitivity: number;
}

export default function Metrics({ fusionHistory, sensitivity }: MetricsProps) {
    const chartData = useMemo(() => {
        return fusionHistory.map((entry, i) => ({
            idx: i,
            time: new Date(entry.time).toLocaleTimeString('en-US', {
                hour12: false,
                minute: '2-digit',
                second: '2-digit',
            }),
            score: parseFloat((entry.score * 100).toFixed(1)),
            threshold: sensitivity * 100,
        }));
    }, [fusionHistory, sensitivity]);

    const latestScore = chartData.length > 0 ? chartData[chartData.length - 1].score : 0;
    const isAboveThreshold = latestScore > sensitivity * 100;

    return (
        <div className="glass-panel-solid glow-border p-4 h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-2 shrink-0">
                <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyber-blue animate-pulse-glow" />
                    <span className="text-[10px] font-mono text-white/30 tracking-widest uppercase">
                        Fusion Probability Timeline
                    </span>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 rounded bg-cyber-blue" />
                        <span className="text-[8px] font-mono text-white/25">SCORE</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-3 h-0.5 rounded bg-scanner-red/50 border-y border-dashed border-scanner-red/30" />
                        <span className="text-[8px] font-mono text-white/25">THRESHOLD</span>
                    </div>
                </div>
            </div>

            {/* Chart */}
            <div className="flex-1 min-h-0">
                {chartData.length > 1 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 5, right: 10, bottom: 0, left: -15 }}>
                            <defs>
                                <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop
                                        offset="5%"
                                        stopColor={isAboveThreshold ? '#EF4444' : '#00d4ff'}
                                        stopOpacity={0.3}
                                    />
                                    <stop
                                        offset="95%"
                                        stopColor={isAboveThreshold ? '#EF4444' : '#00d4ff'}
                                        stopOpacity={0.02}
                                    />
                                </linearGradient>
                            </defs>

                            <CartesianGrid
                                stroke="rgba(255,255,255,0.03)"
                                strokeDasharray="3 6"
                                vertical={false}
                            />

                            <XAxis
                                dataKey="time"
                                tick={{ fill: 'rgba(255,255,255,0.15)', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.05)' }}
                                tickLine={false}
                                interval="preserveStartEnd"
                            />

                            <YAxis
                                domain={[0, 100]}
                                tick={{ fill: 'rgba(255,255,255,0.15)', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                                axisLine={{ stroke: 'rgba(255,255,255,0.05)' }}
                                tickLine={false}
                                tickFormatter={(v: number) => `${v}%`}
                            />

                            <Tooltip
                                contentStyle={{
                                    background: 'rgba(15, 21, 32, 0.95)',
                                    border: '1px solid rgba(255,255,255,0.1)',
                                    borderRadius: '8px',
                                    fontSize: '11px',
                                    fontFamily: 'JetBrains Mono',
                                    color: '#e2e8f0',
                                    padding: '8px 12px',
                                }}
                                labelStyle={{ color: 'rgba(255,255,255,0.4)', fontSize: '9px' }}
                                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Fusion']}
                            />

                            <ReferenceLine
                                y={sensitivity * 100}
                                stroke="#EF4444"
                                strokeDasharray="4 4"
                                strokeOpacity={0.5}
                                label={{
                                    value: 'THRESHOLD',
                                    position: 'insideTopRight',
                                    fill: '#EF444480',
                                    fontSize: 8,
                                    fontFamily: 'JetBrains Mono',
                                }}
                            />

                            <Area
                                type="monotone"
                                dataKey="score"
                                stroke={isAboveThreshold ? '#EF4444' : '#00d4ff'}
                                strokeWidth={2}
                                fill="url(#scoreGradient)"
                                dot={false}
                                activeDot={{
                                    r: 4,
                                    fill: isAboveThreshold ? '#EF4444' : '#00d4ff',
                                    stroke: '#0a0e17',
                                    strokeWidth: 2,
                                }}
                                animationDuration={300}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="flex items-center justify-center h-full">
                        <span className="text-[10px] font-mono text-white/15 tracking-widest">
                            AWAITING DATA...
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}
