/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                'scanner-grey': '#6B7280',
                'scanner-yellow': '#FBBF24',
                'scanner-green': '#10B981',
                'scanner-red': '#EF4444',
                'panel': {
                    900: '#0a0e17',
                    800: '#0f1520',
                    700: '#151d2e',
                    600: '#1a2540',
                    500: '#1e2d4a',
                },
                'cyber': {
                    blue: '#00d4ff',
                    purple: '#7c3aed',
                    teal: '#14b8a6',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
            animation: {
                'scope-spin': 'scopeSpin 8s linear infinite',
                'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
                'scan-line': 'scanLine 3s linear infinite',
                'fade-in': 'fadeIn 0.3s ease-out',
            },
            keyframes: {
                scopeSpin: {
                    '0%': { transform: 'rotate(0deg)' },
                    '100%': { transform: 'rotate(360deg)' },
                },
                pulseGlow: {
                    '0%, 100%': { opacity: '0.4' },
                    '50%': { opacity: '1' },
                },
                scanLine: {
                    '0%': { transform: 'translateY(-100%)' },
                    '100%': { transform: 'translateY(100%)' },
                },
                fadeIn: {
                    '0%': { opacity: '0', transform: 'translateY(4px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
            },
            boxShadow: {
                'glow-green': '0 0 20px rgba(16, 185, 129, 0.3)',
                'glow-red': '0 0 20px rgba(239, 68, 68, 0.3)',
                'glow-yellow': '0 0 20px rgba(251, 191, 36, 0.3)',
                'glow-blue': '0 0 20px rgba(0, 212, 255, 0.2)',
            },
        },
    },
    plugins: [],
}
