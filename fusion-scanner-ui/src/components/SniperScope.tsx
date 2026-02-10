import { useMemo } from 'react'

interface SniperScopeProps {
    size?: number;
    color: string;
    state: string;
    bufferFill: number;
    bufferMax: number;
    confidence: number;
}

export default function SniperScope({
    size = 340,
    color,
    state,
    bufferFill,
    bufferMax,
    confidence,
}: SniperScopeProps) {
    const cx = size / 2;
    const cy = size / 2;
    const bw = 4;
    const cl = 35;
    const radius = (size - 16) / 2;

    // Buffer arc path
    const bufferArc = useMemo(() => {
        if (bufferMax <= 0 || (state !== 'ACQUIRING' && state !== 'REAL' && state !== 'FAKE')) return null;
        const fill = Math.min(bufferFill / bufferMax, 1.0);
        if (fill <= 0) return null;
        return describeArc(cx, cy, radius - 18, -90, -90 + fill * 360);
    }, [cx, cy, radius, bufferFill, bufferMax, state]);

    // Confidence arc
    const confidenceArc = useMemo(() => {
        if (confidence < 0) return null;
        const conf = Math.abs(confidence - 0.5) * 2;
        if (conf <= 0) return null;
        return describeArc(cx, cy, radius - 26, -90, -90 + conf * 360);
    }, [cx, cy, radius, confidence]);

    return (
        <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            className="absolute inset-0 m-auto pointer-events-none"
            style={{ filter: `drop-shadow(0 0 8px ${color}40)` }}
        >
            {/* 1. Outer ring */}
            <circle
                cx={cx} cy={cy} r={radius}
                fill="none"
                stroke={color}
                strokeOpacity={0.15}
                strokeWidth={2}
            />

            {/* 2. Spinning tick marks */}
            <g className="animate-scope-spin" style={{ transformOrigin: `${cx}px ${cy}px` }}>
                {Array.from({ length: 12 }).map((_, i) => {
                    const angle = (i * 30 * Math.PI) / 180;
                    const x1 = cx + (radius - 7) * Math.cos(angle);
                    const y1 = cy + (radius - 7) * Math.sin(angle);
                    const x2 = cx + (radius - 1) * Math.cos(angle);
                    const y2 = cy + (radius - 1) * Math.sin(angle);
                    return (
                        <line
                            key={i}
                            x1={x1} y1={y1} x2={x2} y2={y2}
                            stroke={color}
                            strokeOpacity={0.5}
                            strokeWidth={2}
                        />
                    );
                })}
            </g>

            {/* 3. Corner brackets */}
            <g stroke={color} strokeWidth={bw} fill="none" strokeLinecap="square">
                {/* Top-left */}
                <polyline points={`${bw / 2 + cl},${bw / 2} ${bw / 2},${bw / 2} ${bw / 2},${bw / 2 + cl}`} />
                {/* Top-right */}
                <polyline points={`${size - bw / 2 - cl},${bw / 2} ${size - bw / 2},${bw / 2} ${size - bw / 2},${bw / 2 + cl}`} />
                {/* Bottom-left */}
                <polyline points={`${bw / 2},${size - bw / 2 - cl} ${bw / 2},${size - bw / 2} ${bw / 2 + cl},${size - bw / 2}`} />
                {/* Bottom-right */}
                <polyline points={`${size - bw / 2},${size - bw / 2 - cl} ${size - bw / 2},${size - bw / 2} ${size - bw / 2 - cl},${size - bw / 2}`} />
            </g>

            {/* 4. Crosshair */}
            <g stroke={color} strokeOpacity={0.7} strokeWidth={1}>
                <line x1={cx - 30} y1={cy} x2={cx - 14} y2={cy} />
                <line x1={cx + 14} y1={cy} x2={cx + 30} y2={cy} />
                <line x1={cx} y1={cy - 30} x2={cx} y2={cy - 14} />
                <line x1={cx} y1={cy + 14} x2={cx} y2={cy + 30} />
            </g>
            {/* Center dot */}
            <circle cx={cx} cy={cy} r={2} fill={color} />

            {/* 5. Buffer arc */}
            {bufferArc && (
                <path
                    d={bufferArc}
                    fill="none"
                    stroke={color}
                    strokeOpacity={0.55}
                    strokeWidth={3}
                    strokeLinecap="round"
                />
            )}

            {/* 6. Confidence arc */}
            {confidenceArc && (
                <path
                    d={confidenceArc}
                    fill="none"
                    stroke={color}
                    strokeWidth={4}
                    strokeLinecap="round"
                />
            )}

            {/* Corner dots (decorative) */}
            <circle cx={15} cy={15} r={1.5} fill={color} fillOpacity={0.4} />
            <circle cx={size - 15} cy={15} r={1.5} fill={color} fillOpacity={0.4} />
            <circle cx={15} cy={size - 15} r={1.5} fill={color} fillOpacity={0.4} />
            <circle cx={size - 15} cy={size - 15} r={1.5} fill={color} fillOpacity={0.4} />
        </svg>
    );
}

// ── Helper: SVG arc path ────────────────────────────────────
function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
    const start = polarToCartesian(cx, cy, r, endAngle);
    const end = polarToCartesian(cx, cy, r, startAngle);
    const largeArc = endAngle - startAngle <= 180 ? '0' : '1';
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return {
        x: cx + r * Math.cos(rad),
        y: cy + r * Math.sin(rad),
    };
}
