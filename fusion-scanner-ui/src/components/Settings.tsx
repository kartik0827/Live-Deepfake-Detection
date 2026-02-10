interface SettingsProps {
    autoTracking: boolean;
    sensitivity: number;
    audioPipelineActive: boolean;
    onToggleAutoTracking: () => void;
    onSetSensitivity: (val: number) => void;
    onResetBuffer: () => void;
    onToggleAudio: () => void;
}

export default function Settings({
    autoTracking,
    sensitivity,
    audioPipelineActive,
    onToggleAutoTracking,
    onSetSensitivity,
    onResetBuffer,
    onToggleAudio,
}: SettingsProps) {
    return (
        <div className="glass-panel-solid p-4">
            <div className="flex items-center gap-2 mb-4">
                <svg className="w-3.5 h-3.5 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span className="text-[10px] font-mono text-white/30 tracking-widest uppercase">Control Panel</span>
            </div>

            {/* ── Auto-Tracking Toggle ─────────────────────────── */}
            <div className="flex items-center justify-between mb-4">
                <div>
                    <div className="text-[11px] font-medium text-white/70">Tracking Mode</div>
                    <div className="text-[9px] font-mono text-white/25 mt-0.5">
                        {autoTracking ? 'Auto-tracking active' : 'Manual positioning'}
                    </div>
                </div>
                <button
                    onClick={onToggleAutoTracking}
                    className={`
            relative w-11 h-6 rounded-full transition-all duration-300 focus:outline-none
            ${autoTracking
                            ? 'bg-scanner-green/30 shadow-glow-green'
                            : 'bg-white/10'
                        }
          `}
                >
                    <div
                        className={`
              absolute top-0.5 w-5 h-5 rounded-full transition-all duration-300
              ${autoTracking
                                ? 'left-[22px] bg-scanner-green'
                                : 'left-0.5 bg-scanner-grey'
                            }
            `}
                    />
                </button>
            </div>

            {/* ── Sensitivity Slider ───────────────────────────── */}
            <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                    <div className="text-[11px] font-medium text-white/70">Sensitivity Threshold</div>
                    <span className="text-[11px] font-mono text-cyber-blue/60 tabular-nums">
                        {sensitivity.toFixed(2)}
                    </span>
                </div>
                <div className="relative">
                    <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.01}
                        value={sensitivity}
                        onChange={(e) => onSetSensitivity(parseFloat(e.target.value))}
                        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-panel-600
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:w-3.5
              [&::-webkit-slider-thumb]:h-3.5
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-cyber-blue
              [&::-webkit-slider-thumb]:shadow-glow-blue
              [&::-webkit-slider-thumb]:cursor-pointer
              [&::-webkit-slider-thumb]:transition-transform
              [&::-webkit-slider-thumb]:hover:scale-125
            "
                    />
                    <div className="flex justify-between mt-1 text-[8px] font-mono text-white/15">
                        <span>0.0</span>
                        <span>0.5</span>
                        <span>1.0</span>
                    </div>
                </div>
            </div>

            {/* ── Action Buttons ───────────────────────────────── */}
            <div className="flex gap-2">
                <button
                    onClick={onResetBuffer}
                    className="flex-1 px-3 py-2 rounded-lg text-[10px] font-mono font-semibold tracking-wider uppercase
            bg-panel-600/80 text-white/50 border border-white/5
            hover:bg-panel-500 hover:text-white/70 hover:border-white/10
            active:scale-[0.97] transition-all duration-200
          "
                >
                    Reset Buffer
                </button>

                <button
                    onClick={onToggleAudio}
                    className={`flex-1 px-3 py-2 rounded-lg text-[10px] font-mono font-semibold tracking-wider uppercase
            border transition-all duration-200 active:scale-[0.97]
            ${audioPipelineActive
                            ? 'bg-cyber-purple/15 text-cyber-purple/70 border-cyber-purple/20 hover:bg-cyber-purple/25'
                            : 'bg-panel-600/80 text-white/30 border-white/5 hover:bg-panel-500'
                        }
          `}
                >
                    {audioPipelineActive ? '🔊 Audio ON' : '🔇 Audio OFF'}
                </button>
            </div>
        </div>
    );
}
