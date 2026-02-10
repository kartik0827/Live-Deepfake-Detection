import { useRef, useEffect, useState } from 'react'
import SniperScope from './SniperScope'
import type { ScannerData } from '../App'

interface VideoFeedProps {
    data: ScannerData;
    videoFrame: string | null;
}

const STATE_COLORS: Record<string, string> = {
    SEARCHING: '#6B7280',
    ACQUIRING: '#FBBF24',
    REAL: '#10B981',
    FAKE: '#EF4444',
};

export default function VideoFeed({ data, videoFrame }: VideoFeedProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ w: 640, h: 480 });

    // ── Resize observer ─────────────────────────────────────
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;

        const observer = new ResizeObserver(([entry]) => {
            const { width, height } = entry.contentRect;
            setDimensions({ w: Math.floor(width), h: Math.floor(height) });
        });
        observer.observe(el);
        return () => observer.disconnect();
    }, []);

    // ── Draw video frame to canvas ──────────────────────────
    useEffect(() => {
        if (!videoFrame || !canvasRef.current) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, dimensions.w, dimensions.h);
        };
        img.src = `data:image/jpeg;base64,${videoFrame}`;
    }, [videoFrame, dimensions]);

    // ── Draw demo pattern when no video ─────────────────────
    useEffect(() => {
        if (videoFrame || !canvasRef.current) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        const { w, h } = dimensions;

        // Dark background with subtle gradient
        const grad = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) / 2);
        grad.addColorStop(0, '#111827');
        grad.addColorStop(1, '#0a0e17');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Grid lines
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.04)';
        ctx.lineWidth = 0.5;
        const gridSize = 30;
        for (let x = 0; x < w; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Center text
        ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
        ctx.font = '600 14px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('VIDEO FEED', w / 2, h / 2 - 10);
        ctx.font = '400 10px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.fillText('Connect scanner to begin analysis', w / 2, h / 2 + 10);
    }, [videoFrame, dimensions]);

    const scopeSize = Math.min(dimensions.w, dimensions.h) * 0.65;
    const color = STATE_COLORS[data.state] || STATE_COLORS.SEARCHING;

    return (
        <div
            ref={containerRef}
            className="relative w-full h-full glass-panel glow-border overflow-hidden"
        >
            {/* Canvas for video / demo pattern */}
            <canvas
                ref={canvasRef}
                width={dimensions.w}
                height={dimensions.h}
                className="absolute inset-0 w-full h-full"
            />

            {/* Grid overlay */}
            <div className="absolute inset-0 grid-overlay pointer-events-none" />

            {/* Scan line effect */}
            <div className="scan-line-effect" />

            {/* Sniper scope overlay */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <SniperScope
                    size={scopeSize}
                    color={color}
                    state={data.state}
                    bufferFill={data.bufferFill}
                    bufferMax={data.bufferMax}
                    confidence={data.fusionScore}
                />
            </div>

            {/* Top-left info overlay */}
            <div className="absolute top-3 left-3 flex flex-col gap-1">
                <div className="flex items-center gap-2 bg-black/50 backdrop-blur-sm rounded px-2 py-0.5">
                    <div className={`w-1.5 h-1.5 rounded-full ${data.state === 'SEARCHING' ? 'bg-scanner-grey animate-pulse' : 'bg-scanner-green'}`} />
                    <span className="text-[9px] font-mono text-white/50 tracking-wider">
                        {data.autoTracking ? 'AUTO-TRACK' : 'MANUAL'}
                    </span>
                </div>
                <div className="bg-black/50 backdrop-blur-sm rounded px-2 py-0.5">
                    <span className="text-[9px] font-mono text-white/30">
                        BUF {data.bufferFill}/{data.bufferMax}
                    </span>
                </div>
            </div>

            {/* Bottom center: fusion score */}
            {data.fusionScore > 0 && (
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2">
                    <div
                        className="px-4 py-1.5 rounded-lg border backdrop-blur-sm font-mono text-sm font-bold tracking-wider transition-all duration-300"
                        style={{
                            backgroundColor: `${color}20`,
                            borderColor: `${color}40`,
                            color: color,
                            boxShadow: `0 0 20px ${color}25`,
                        }}
                    >
                        FUSION: {data.state === 'FAKE' || data.state === 'REAL'
                            ? `${data.state} ${Math.round(
                                data.state === 'FAKE'
                                    ? data.fusionScore * 100
                                    : (1 - data.fusionScore) * 100
                            )}%`
                            : `${Math.round(data.fusionScore * 100)}%`}
                    </div>
                </div>
            )}
        </div>
    );
}
