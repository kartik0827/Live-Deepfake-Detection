const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let mainWindow = null;
let pythonProcess = null;

// ─── Window Creation ────────────────────────────────────────────
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1100,
        minHeight: 700,
        backgroundColor: '#0a0e17',
        titleBarStyle: 'hidden',
        titleBarOverlay: {
            color: '#0a0e17',
            symbolColor: '#6B7280',
            height: 36,
        },
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    // In development, load Vite dev server; in production, load built files
    const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
    } else {
        mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// ─── Python Child Process ───────────────────────────────────────
function spawnPython() {
    const bridgePath = path.join(__dirname, '..', 'bridge.py');
    const projectRoot = path.join(__dirname, '..', '..');

    // Try 'python' first, fall back to 'python3'
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

    pythonProcess = spawn(pythonCmd, [bridgePath], {
        cwd: projectRoot,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let buffer = '';

    // Read JSON lines from Python stdout
    pythonProcess.stdout.on('data', (data) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete line in buffer

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            try {
                const parsed = JSON.parse(trimmed);
                if (mainWindow && !mainWindow.isDestroyed()) {
                    // Route messages by type
                    switch (parsed.type) {
                        case 'detection':
                            mainWindow.webContents.send('detection-data', parsed);
                            break;
                        case 'video_frame':
                            mainWindow.webContents.send('video-frame', parsed);
                            break;
                        case 'status':
                            mainWindow.webContents.send('status-update', parsed);
                            break;
                        default:
                            mainWindow.webContents.send('detection-data', parsed);
                    }
                }
            } catch (e) {
                // Non-JSON output → log for debugging
                console.log('[Python]', trimmed);
            }
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error('[Python ERR]', data.toString());
    });

    pythonProcess.on('close', (code) => {
        console.log(`[Python] Process exited with code ${code}`);
        pythonProcess = null;
    });

    pythonProcess.on('error', (err) => {
        console.error('[Python] Failed to start:', err.message);
        pythonProcess = null;
    });
}

// ─── IPC Handlers ───────────────────────────────────────────────
ipcMain.on('send-command', (event, command) => {
    if (pythonProcess && pythonProcess.stdin.writable) {
        const msg = JSON.stringify(command) + '\n';
        pythonProcess.stdin.write(msg);
    }
});

ipcMain.handle('get-status', () => {
    return {
        pythonRunning: pythonProcess !== null && !pythonProcess.killed,
        windowId: mainWindow ? mainWindow.id : null,
    };
});

ipcMain.on('start-scanner', () => {
    if (!pythonProcess) {
        spawnPython();
    }
});

ipcMain.on('stop-scanner', () => {
    if (pythonProcess) {
        pythonProcess.stdin.write(JSON.stringify({ command: 'quit' }) + '\n');
        setTimeout(() => {
            if (pythonProcess && !pythonProcess.killed) {
                pythonProcess.kill();
            }
        }, 3000);
    }
});

// ─── App Lifecycle ──────────────────────────────────────────────
app.whenReady().then(() => {
    createWindow();
    // Don't auto-spawn Python; let the user start it from the UI
});

app.on('window-all-closed', () => {
    if (pythonProcess && !pythonProcess.killed) {
        pythonProcess.kill();
    }
    app.quit();
});

app.on('before-quit', () => {
    if (pythonProcess && !pythonProcess.killed) {
        pythonProcess.kill();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
