import './style.css';
import { CellularAutomata } from './ca';

// Get DOM elements
const canvas = document.getElementById('caCanvas') as HTMLCanvasElement;
const statusDiv = document.getElementById('status') as HTMLElement;
const pauseBtn = document.getElementById('pauseBtn') as HTMLButtonElement;
const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
const emojiButtonsContainer = document.getElementById('emojiButtons') as HTMLElement;
const speedSlider = document.getElementById('speedSlider') as HTMLInputElement;
const speedValue = document.getElementById('speedValue') as HTMLElement;

// Initialize CA
const ca = new CellularAutomata(canvas, statusDiv);

// Button handlers
pauseBtn.addEventListener('click', () => {
  const isRunning = ca.togglePause();
  pauseBtn.textContent = isRunning ? 'Pause' : 'Resume';
});

resetBtn.addEventListener('click', () => {
  ca.reset();
});

// Speed slider handler
// Slider range: 0 to 100
// 0 = 1/60x (1 step/sec)
// 50 = 1x (60 steps/sec)
// 100 = 60x (3600 steps/sec)
// Use logarithmic scale
function sliderToSpeed(sliderValue: number): number {
  // Map 0-100 slider to 1-3600 steps/sec logarithmically
  // log(1) = 0, log(60) = 4.09, log(3600) = 8.19
  // Slider 0 -> 1 step/sec
  // Slider 50 -> 60 steps/sec
  // Slider 100 -> 3600 steps/sec

  const minLog = Math.log(1);
  const maxLog = Math.log(3600);
  const scale = (maxLog - minLog) / 100;

  return Math.exp(minLog + scale * sliderValue);
}

function formatSpeed(stepsPerSecond: number): string {
  const multiplier = stepsPerSecond / 60;
  if (multiplier < 0.1) {
    return `1/${Math.round(1/multiplier)}x`;
  } else if (multiplier < 10) {
    return `${multiplier.toFixed(1)}x`;
  } else {
    return `${Math.round(multiplier)}x`;
  }
}

speedSlider.addEventListener('input', (e) => {
  const target = e.target as HTMLInputElement;
  const speed = sliderToSpeed(parseFloat(target.value));
  ca.setSpeed(speed);
  speedValue.textContent = formatSpeed(speed);
});

// Initialize speed to 1x (60 steps per second)
ca.setSpeed(60);
speedValue.textContent = '1.0x';

// Create emoji selector buttons
function createEmojiButtons(): void {
  for (let i = 0; i < 10; i++) {
    const btn = document.createElement('button');
    btn.className = 'emoji-btn';
    btn.dataset.emoji = i.toString();
    btn.title = `Emoji ${i}`;

    const img = document.createElement('img');
    img.src = `/emojis/emoji_${i}.png`;
    img.alt = `Emoji ${i}`;

    btn.appendChild(img);
    btn.addEventListener('click', () => loadModel(i));
    emojiButtonsContainer.appendChild(btn);
  }
}

// Load a specific emoji model
async function loadModel(emojiNum: number): Promise<void> {
  await ca.loadModel(emojiNum);

  // Update button states
  document.querySelectorAll('.emoji-btn').forEach((btn) => {
    const btnElement = btn as HTMLButtonElement;
    btnElement.classList.toggle(
      'active',
      parseInt(btnElement.dataset.emoji || '-1') === emojiNum
    );
  });
}

// Initialize
function init(): void {
  createEmojiButtons();
  // Load emoji 0 by default
  loadModel(0);
}

// Start when page loads
init();
