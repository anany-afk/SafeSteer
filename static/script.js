document.addEventListener('DOMContentLoaded', () => {

    const els = {
        masterStatus: document.getElementById('masterStatus'),
        alertPlate: document.getElementById('alertPlate'),
        alertText: document.getElementById('alertText'),
        statBlinkRate: document.getElementById('stat-blinkRate'),
        statEyeClose: document.getElementById('stat-eyeClose'),
        statScore: document.getElementById('stat-drowsyScore'),
        statYawns: document.getElementById('stat-yawns'),
        statLighting: document.getElementById('stat-lighting'),
        dangerOverlay: document.getElementById('dangerOverlay'),
        soundToggle: document.getElementById('soundToggle')
    };

    // Chart.js Setup
    const ctx = document.getElementById('fatigueChart').getContext('2d');
    let dbData = [];
    let dbLabels = [];
    const maxDataPoints = 30;

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dbLabels,
            datasets: [{
                label: 'Drowsiness Score',
                data: dbData,
                borderColor: '#00E5FF',
                backgroundColor: 'rgba(0, 229, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false, min: 0, max: 100 }
            },
            animation: { duration: 0 }
        }
    });

    let isBeeping = false;

    // Simulate Beep
    const playBeep = () => {
        if (!els.soundToggle || !els.soundToggle.checked || isBeeping) return;
        isBeeping = true;
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        osc.frequency.value = 800; // hz
        osc.connect(audioCtx.destination);
        osc.start();
        setTimeout(() => { osc.stop(); isBeeping = false; }, 200);
    };

    if (els.soundToggle) {
        els.soundToggle.addEventListener('change', (e) => {
            if (!e.target.checked) {
                alert("WARNING: Sound alerts are disabled. Please remain attentive while driving without audio warnings.");
            }
        });
    }

    const updateDashboard = async () => {
        try {
            const response = await fetch('/stats');
            const data = await response.json();

            if (data.status === "starting") return;

            // Derived stats
            const currentProb = data.prob || 0;
            const score = Math.round(currentProb * 100);
            
            // Blink rate = blinks / elapsed minutes
            const elapsedMins = (data.elapsed || 1) / 60;
            const blinkRate = Math.round((data.blinks || 0) / Math.max(0.1, elapsedMins));

            // Populate cards
            els.statBlinkRate.textContent = blinkRate;
            
            // Eye closure duration ~ based on perclos and elapsed.
            const closureDur = (data.perclos * 4).toFixed(1); 
            els.statEyeClose.textContent = closureDur;
            els.statScore.textContent = score;
            els.statYawns.textContent = data.yawns || 0;
            els.statLighting.textContent = "Normal";

            const isDrowsy = data.status.includes('DROWSY');

            if (isDrowsy) {
                els.masterStatus.textContent = "DROWSY";
                els.masterStatus.className = "status-drowsy";
                els.alertPlate.classList.add('active-alert');
                els.alertText.textContent = "⚠️ Drowsiness Detected!";
                els.dangerOverlay.classList.add('active');
                playBeep();
            } else {
                els.masterStatus.textContent = "AWAKE";
                els.masterStatus.className = "status-awake";
                els.alertPlate.classList.remove('active-alert');
                els.alertText.textContent = "System Normal";
                els.dangerOverlay.classList.remove('active');
            }

            // Update Graph
            if (dbData.length > maxDataPoints) {
                dbData.shift();
                dbLabels.shift();
            }
            dbData.push(score);
            dbLabels.push('');
            
            // Change line color if drowsy
            chart.data.datasets[0].borderColor = isDrowsy ? '#FF3B3B' : '#00E5FF';
            chart.data.datasets[0].backgroundColor = isDrowsy ? 'rgba(255, 59, 59, 0.1)' : 'rgba(0, 229, 255, 0.1)';
            
            chart.update();

        } catch (e) {
            console.error("Fetch stats error:", e);
        }
    };

    setInterval(updateDashboard, 200);
});
