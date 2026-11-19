/**
 * Global Application Utilities and Topbar Synchronization
 */

document.addEventListener('DOMContentLoaded', () => {
    syncTopbarChillerStatus();
    setInterval(syncTopbarChillerStatus, 15000);
});

async function syncTopbarChillerStatus() {
    try {
        const res = await fetch('/api/cooling/status');
        const json = await res.json();
        if (!json.success || !json.data) return;

        const d = json.data;
        const pill = document.getElementById('navChillerStatus');
        const dot = document.getElementById('navChillerDot');
        const text = document.getElementById('navChillerText');
        
        const isRunning = (d.is_running || d.chiller_state === 'ON' || d.chiller_state === 'RUNNING' || d.canonical_chiller_state === 'ON');
        if (pill && text && dot) {
            if (isRunning) {
                pill.className = 'status-pill-lg running';
                dot.className = 'dot-indicator green';
                text.textContent = 'CHILLER ACTIVE (ON)';
            } else {
                pill.className = 'status-pill-lg stopped';
                dot.className = 'dot-indicator gray';
                text.textContent = 'CHILLER IDLE (OFF)';
            }
        }
    } catch (e) {
        // Silent catch for background status poll
    }
}
