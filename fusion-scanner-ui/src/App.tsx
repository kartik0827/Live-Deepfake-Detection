import { useState, useEffect, useCallback } from 'react'
import Dashboard from './components/Dashboard'
import Sidebar from './components/Sidebar'
import type { DetectionData, VideoFrameData, StatusData } from './vite-env'

export type ScannerState = 'SEARCHING' | 'ACQUIRING' | 'REAL' | 'FAKE';

export interface ScannerData {
    visualScore: number;
    audioScore: number;
    fusionScore: number;
    state: ScannerState;
    bufferFill: number;
    bufferMax: number;
    fusionHistory: { time: number; score: number }[];
    isConnected: boolean;
    autoTracking: boolean;
    sensitivity: number;
    audioPipelineActive: boolean;
}

const initialData: ScannerData = {
    visualScore: 0,
    audioScore: 0,
    fusionScore: 0,
    state: 'SEARCHING',
    bufferFill: 0,
    bufferMax: 10,
    fusionHistory: [],
    isConnected: false,
    autoTracking: true,
    sensitivity: 0.5,
    audioPipelineActive: true,
};

export default function App() {
    const [data, setData] = useState<ScannerData>(initialData);
    const [videoFrame, setVideoFrame] = useState<string | null>(null);

    // ── IPC Listeners ─────────────────────────────────────────
    useEffect(() => {
        const api = window.electronAPI;
        if (!api) return;

        api.onDetectionData((d: DetectionData) => {
            setData(prev => {
                const fusionScore = d.fusion_score;
                let state: ScannerState = 'SEARCHING';
                if (d.state === 'SEARCHING') state = 'SEARCHING';
                else if (d.state === 'ACQUIRING') state = 'ACQUIRING';
                else if (d.label === 'REAL') state = 'REAL';
                else if (d.label === 'FAKE') state = 'FAKE';

                const newEntry = { time: Date.now(), score: fusionScore };
                const history = [...prev.fusionHistory, newEntry].slice(-60);

                return {
                    ...prev,
                    visualScore: d.visual_score,
                    audioScore: d.audio_score,
                    fusionScore,
                    state,
                    bufferFill: d.buffer_fill,
                    bufferMax: d.buffer_max,
                    fusionHistory: history,
                    isConnected: true,
                };
            });
        });

        api.onVideoFrame((d: VideoFrameData) => {
            setVideoFrame(d.frame);
        });

        api.onStatusUpdate((d: StatusData) => {
            console.log(`[Status] ${d.level}: ${d.message}`);
        });

        return () => {
            api.removeAllListeners('detection-data');
            api.removeAllListeners('video-frame');
            api.removeAllListeners('status-update');
        };
    }, []);

    // ── Simulated demo data (when Python is not connected) ────
    useEffect(() => {
        if (data.isConnected) return;

        const interval = setInterval(() => {
            setData(prev => {
                // Cycle through states for demo
                const cycle = Date.now() % 20000;
                let state: ScannerState = 'SEARCHING';
                let visual = 0;
                let audio = 0;

                if (cycle < 4000) {
                    state = 'SEARCHING';
                    visual = 0;
                    audio = 0;
                } else if (cycle < 7000) {
                    state = 'ACQUIRING';
                    visual = Math.random() * 0.3;
                    audio = Math.random() * 0.2;
                } else if (cycle < 14000) {
                    state = 'REAL';
                    visual = 0.15 + Math.random() * 0.2;
                    audio = 0.1 + Math.random() * 0.15;
                } else {
                    state = 'FAKE';
                    visual = 0.65 + Math.random() * 0.3;
                    audio = 0.55 + Math.random() * 0.35;
                }

                const fusion = visual * 0.7 + audio * 0.3;
                const bufferFill = state === 'SEARCHING' ? 0 :
                    state === 'ACQUIRING' ? Math.floor(Math.random() * 7) :
                        10;

                const newEntry = { time: Date.now(), score: fusion };
                const history = [...prev.fusionHistory, newEntry].slice(-60);

                return {
                    ...prev,
                    visualScore: visual,
                    audioScore: audio,
                    fusionScore: fusion,
                    state,
                    bufferFill,
                    fusionHistory: history,
                };
            });
        }, 500);

        return () => clearInterval(interval);
    }, [data.isConnected]);

    // ── Commands ──────────────────────────────────────────────
    const sendCommand = useCallback((cmd: string, params: Record<string, unknown> = {}) => {
        const api = window.electronAPI;
        if (api) {
            api.sendCommand({ command: cmd, ...params });
        }
    }, []);

    const toggleAutoTracking = useCallback(() => {
        setData(prev => {
            const next = !prev.autoTracking;
            sendCommand('set_mode', { mode: next ? 'auto' : 'manual' });
            return { ...prev, autoTracking: next };
        });
    }, [sendCommand]);

    const setSensitivity = useCallback((value: number) => {
        setData(prev => ({ ...prev, sensitivity: value }));
        sendCommand('set_threshold', { threshold: value });
    }, [sendCommand]);

    const resetBuffer = useCallback(() => {
        sendCommand('reset_buffer');
        setData(prev => ({ ...prev, bufferFill: 0, fusionScore: 0, state: 'ACQUIRING' as ScannerState }));
    }, [sendCommand]);

    const toggleAudio = useCallback(() => {
        setData(prev => {
            const next = !prev.audioPipelineActive;
            sendCommand('toggle_audio', { enabled: next });
            return { ...prev, audioPipelineActive: next };
        });
    }, [sendCommand]);

    return (
        <div className="flex h-screen w-screen overflow-hidden bg-panel-900">
            {/* Titlebar drag area */}
            <div className="titlebar-drag fixed top-0 left-0 right-0 h-9 z-50" />

            {/* Main layout */}
            <div className="flex w-full h-full pt-9">
                {/* Main content */}
                <div className="flex-1 min-w-0 flex flex-col">
                    <Dashboard
                        data={data}
                        videoFrame={videoFrame}
                    />
                </div>

                {/* Sidebar */}
                <Sidebar
                    data={data}
                    onToggleAutoTracking={toggleAutoTracking}
                    onSetSensitivity={setSensitivity}
                    onResetBuffer={resetBuffer}
                    onToggleAudio={toggleAudio}
                />
            </div>
        </div>
    );
}
