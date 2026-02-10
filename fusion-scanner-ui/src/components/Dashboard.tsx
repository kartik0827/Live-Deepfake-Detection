import VideoFeed from './VideoFeed'
import Metrics from './Metrics'
import type { ScannerData } from '../App'

interface DashboardProps {
    data: ScannerData;
    videoFrame: string | null;
}

export default function Dashboard({ data, videoFrame }: DashboardProps) {
    return (
        <div className="flex-1 flex flex-col gap-4 p-4 overflow-hidden">
            {/* ── Header bar ──────────────────────────────────────── */}
            <div className="flex items-center justify-between shrink-0">
                <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-cyber-blue animate-pulse-glow" />
                    <h1 className="text-lg font-bold tracking-wide text-white/90">
                        FUSION<span className="text-cyber-blue">SCANNER</span>
                    </h1>
                    <span className="text-[10px] font-mono text-white/30 tracking-widest uppercase ml-1">
                        Live Deepfake Detection
                    </span>
                </div>

                <div className="flex items-center gap-4">
                    {/* Connection indicator */}
                    <div className="flex items-center gap-2">
                        <div className={`w-1.5 h-1.5 rounded-full ${data.isConnected ? 'bg-scanner-green' : 'bg-scanner-grey'}`} />
                        <span className="text-[10px] font-mono text-white/40">
                            {data.isConnected ? 'PIPELINE ACTIVE' : 'DEMO MODE'}
                        </span>
                    </div>

                    {/* State badge */}
                    <StatusBadge state={data.state} />
                </div>
            </div>

            {/* ── Video Feed (main area) ──────────────────────────── */}
            <div className="flex-1 min-h-0">
                <VideoFeed
                    data={data}
                    videoFrame={videoFrame}
                />
            </div>

            {/* ── Metrics Chart (bottom strip) ────────────────────── */}
            <div className="shrink-0 h-[180px]">
                <Metrics
                    fusionHistory={data.fusionHistory}
                    sensitivity={data.sensitivity}
                />
            </div>
        </div>
    );
}

function StatusBadge({ state }: { state: string }) {
    const config: Record<string, { label: string; cls: string }> = {
        SEARCHING: { label: 'SEARCHING', cls: 'status-searching' },
        ACQUIRING: { label: 'ACQUIRING', cls: 'status-acquiring' },
        REAL: { label: 'REAL', cls: 'status-real' },
        FAKE: { label: '⚠ FAKE', cls: 'status-fake' },
    };

    const { label, cls } = config[state] || config.SEARCHING;

    return (
        <div className={`px-3 py-1 rounded-md border text-xs font-bold font-mono tracking-wider ${cls} transition-all duration-300`}>
            {label}
        </div>
    );
}
