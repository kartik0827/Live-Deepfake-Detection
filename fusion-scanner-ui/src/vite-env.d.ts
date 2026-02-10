/// <reference types="vite/client" />

// Type definitions for the Electron IPC API exposed via preload.js
export interface ElectronAPI {
    onDetectionData: (callback: (data: DetectionData) => void) => void;
    onVideoFrame: (callback: (data: VideoFrameData) => void) => void;
    onStatusUpdate: (callback: (data: StatusData) => void) => void;
    sendCommand: (command: BridgeCommand) => void;
    startScanner: () => void;
    stopScanner: () => void;
    getStatus: () => Promise<{ pythonRunning: boolean; windowId: number | null }>;
    removeAllListeners: (channel: string) => void;
}

export interface DetectionData {
    type: 'detection';
    visual_score: number;
    audio_score: number;
    fusion_score: number;
    state: 'SEARCHING' | 'ACQUIRING' | 'LOCKED';
    label: 'REAL' | 'FAKE' | 'UNKNOWN';
    buffer_fill: number;
    buffer_max: number;
    timestamp: number;
}

export interface VideoFrameData {
    type: 'video_frame';
    frame: string;
    width: number;
    height: number;
}

export interface StatusData {
    type: 'status';
    message: string;
    level: 'info' | 'warn' | 'error';
}

export interface BridgeCommand {
    command: string;
    [key: string]: unknown;
}

declare global {
    interface Window {
        electronAPI?: ElectronAPI;
    }
}
