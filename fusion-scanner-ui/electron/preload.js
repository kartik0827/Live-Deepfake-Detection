const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // ─── Receive data from Python ───────────────────────────────
    onDetectionData: (callback) => {
        ipcRenderer.on('detection-data', (_event, data) => callback(data));
    },

    onVideoFrame: (callback) => {
        ipcRenderer.on('video-frame', (_event, data) => callback(data));
    },

    onStatusUpdate: (callback) => {
        ipcRenderer.on('status-update', (_event, data) => callback(data));
    },

    // ─── Send commands to Python ────────────────────────────────
    sendCommand: (command) => {
        ipcRenderer.send('send-command', command);
    },

    // ─── Scanner lifecycle ──────────────────────────────────────
    startScanner: () => {
        ipcRenderer.send('start-scanner');
    },

    stopScanner: () => {
        ipcRenderer.send('stop-scanner');
    },

    // ─── Status query ───────────────────────────────────────────
    getStatus: () => {
        return ipcRenderer.invoke('get-status');
    },

    // ─── Cleanup ────────────────────────────────────────────────
    removeAllListeners: (channel) => {
        ipcRenderer.removeAllListeners(channel);
    },
});
