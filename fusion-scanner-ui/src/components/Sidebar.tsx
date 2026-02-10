import Settings from './Settings'
import type { ScannerData } from '../App'

interface SidebarProps {
    data: ScannerData;
    onToggleAutoTracking: () => void;
    onSetSensitivity: (val: number) => void;
    onResetBuffer: () => void;
    onToggleAudio: () => void;
}

const STATE_INFO: Record<string, { label: string; cls: string; desc: string }> = {
    SEARCHING: { label: 'SEARCHING', cls: 'status-searching', desc: 'Scanning for faces...' },
    ACQUIRING: { label: 'ACQUIRING', cls: 'status-acquiring', desc: 'Face detected, buffering...' },
    REAL: { label: 'REAL', cls: 'status-real', desc: 'Authentic source detected' },
    FAKE: { label: '⚠ FAKE', cls: 'status-fake', desc: 'Deepfake anomaly detected' },
};

export default function Sidebar({
    data,
    onToggleAutoTracking,
    onSetSensitivity,
    onResetBuffer,
    onToggleAudio,
}: SidebarProps) {
    const info = STATE_INFO[data.state] || STATE_INFO.SEARCHING;

    return (
        <div className="w-[320px] shrink-0 flex flex-col gap-3 p-3 border-l border-white/5 bg-panel-800/40 overflow-y-auto no-drag">

            {/* ── Status Card ──────────────────────────────────────── */}
            <div className="glass-panel-solid p-4 glow-border">
                <div className="flex items-center justify-between mb-3">
                    <span className="text-[10px] font-mono text-white/30 tracking-widest uppercase">Detection Status</span>
                    <div className={`px-2.5 py-0.5 rounded border text-[11px] font-bold font-mono tracking-wider ${info.cls} transition-all duration-300`}>
                        {info.label}
                    </div>
                </div>
                <p className="text-[11px] text-white/40 font-mono">{info.desc}</p>

                {/* Buffer progress */}
                <div className="mt-3">
                    <div className="flex justify-between text-[9px] font-mono text-white/30 mb-1">
                        <span>FRAME BUFFER</span>
                        <span>{data.bufferFill}/{data.bufferMax}</span>
                    </div>
                    <div className="h-1.5 rounded-full gauge-track overflow-hidden">
                        <div
                            className="h-full rounded-full transition-all duration-300 gauge-fill-yellow"
                            style={{ width: `${(data.bufferFill / data.bufferMax) * 100}%` }}
                        />
                    </div>
                </div>
            </div>

            {/* ── Visual Score ─────────────────────────────────────── */}
            <ScoreCard
                label="VISUAL SCORE"
                weight="70% weight"
                score={data.visualScore}
                isActive={data.state !== 'SEARCHING'}
            />

            {/* ── Audio Score ──────────────────────────────────────── */}
            <ScoreCard
                label="AUDIO SCORE"
                weight="30% weight"
                score={data.audioScore}
                isActive={data.audioPipelineActive && data.state !== 'SEARCHING'}
                disabled={!data.audioPipelineActive}
            />

            {/* ── Fusion Score ─────────────────────────────────────── */}
            <div className="glass-panel-solid p-4">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-mono text-white/30 tracking-widest uppercase">Fusion Score</span>
                    <span className="text-[10px] font-mono text-cyber-blue/60">V×0.7 + A×0.3</span>
                </div>
                <div className="flex items-end gap-2">
                    <span
                        className="text-3xl font-bold font-mono tabular-nums transition-colors duration-300"
                        style={{ color: getFusionColor(data.fusionScore, data.sensitivity) }}
                    >
                        {data.state === 'SEARCHING' ? '---' : (data.fusionScore * 100).toFixed(1)}
                    </span>
                    {data.state !== 'SEARCHING' && (
                        <span className="text-sm font-mono text-white/30 mb-0.5">%</span>
                    )}
                </div>
                <div className="mt-2 h-2 rounded-full gauge-track overflow-hidden">
                    <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                            width: `${data.fusionScore * 100}%`,
                            background: `linear-gradient(90deg, #10B981, ${data.fusionScore > data.sensitivity ? '#EF4444' : '#10B981'})`,
                        }}
                    />
                </div>
                {/* Threshold marker */}
                <div className="relative h-0">
                    <div
                        className="absolute -top-2 w-0.5 h-2 bg-white/40"
                        style={{ left: `${data.sensitivity * 100}%` }}
                    />
                </div>
                <div className="flex justify-between mt-2 text-[8px] font-mono text-white/20">
                    <span>REAL</span>
                    <span>THRESHOLD {(data.sensitivity * 100).toFixed(0)}%</span>
                    <span>FAKE</span>
                </div>
            </div>

            {/* ── Settings / Control Panel ─────────────────────────── */}
            <Settings
                autoTracking={data.autoTracking}
                sensitivity={data.sensitivity}
                audioPipelineActive={data.audioPipelineActive}
                onToggleAutoTracking={onToggleAutoTracking}
                onSetSensitivity={onSetSensitivity}
                onResetBuffer={onResetBuffer}
                onToggleAudio={onToggleAudio}
            />

            {/* ── Footer ───────────────────────────────────────────── */}
            <div className="mt-auto pt-2 border-t border-white/5">
                <div className="flex items-center justify-between px-1">
                    <span className="text-[8px] font-mono text-white/15 tracking-widest">FUSION SCANNER v1.0</span>
                    <span className="text-[8px] font-mono text-white/15">
                        {new Date().toLocaleTimeString('en-US', { hour12: false })}
                    </span>
                </div>
            </div>
        </div>
    );
}

// ── Score Card Component ────────────────────────────────────
function ScoreCard({
    label,
    weight,
    score,
    isActive,
    disabled = false,
}: {
    label: string;
    weight: string;
    score: number;
    isActive: boolean;
    disabled?: boolean;
}) {
    const percentage = Math.round(score * 100);
    const isHighRisk = score > 0.5;
    const fillClass = isHighRisk ? 'gauge-fill-red' : 'gauge-fill-green';

    return (
        <div className={`glass-panel-solid p-4 transition-opacity duration-300 ${disabled ? 'opacity-40' : ''}`}>
            <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono text-white/30 tracking-widest uppercase">{label}</span>
                <span className="text-[9px] font-mono text-cyber-blue/40">{weight}</span>
            </div>

            <div className="flex items-center gap-3">
                {/* Score number */}
                <span
                    className="text-2xl font-bold font-mono tabular-nums transition-colors duration-300"
                    style={{ color: !isActive ? '#4B5563' : isHighRisk ? '#EF4444' : '#10B981' }}
                >
                    {!isActive ? '--' : percentage}
                </span>
                {isActive && <span className="text-xs font-mono text-white/20">%</span>}

                {/* Gauge bar */}
                <div className="flex-1 h-1.5 rounded-full gauge-track overflow-hidden">
                    <div
                        className={`h-full rounded-full transition-all duration-500 ${fillClass}`}
                        style={{ width: isActive ? `${percentage}%` : '0%' }}
                    />
                </div>
            </div>

            {/* Risk label */}
            {isActive && (
                <div className="mt-1.5 flex items-center gap-1.5">
                    <div
                        className="w-1 h-1 rounded-full"
                        style={{ backgroundColor: isHighRisk ? '#EF4444' : '#10B981' }}
                    />
                    <span
                        className="text-[9px] font-mono tracking-wider"
                        style={{ color: isHighRisk ? '#EF4444' : '#10B981' }}
                    >
                        {isHighRisk ? 'HIGH RISK' : 'LOW RISK'}
                    </span>
                </div>
            )}
        </div>
    );
}

function getFusionColor(score: number, threshold: number): string {
    if (score <= 0) return '#4B5563';
    return score > threshold ? '#EF4444' : '#10B981';
}
