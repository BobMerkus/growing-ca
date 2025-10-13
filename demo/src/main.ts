import './style.css';
import { CellularAutomata } from './ca';

// Get DOM elements
const canvas = document.getElementById('caCanvas') as HTMLCanvasElement;
const statusDiv = document.getElementById('status') as HTMLElement;
const pauseBtn = document.getElementById('pauseBtn') as HTMLButtonElement;
const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
const emojiButtonsContainer = document.getElementById('emojiButtons') as HTMLElement;

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
