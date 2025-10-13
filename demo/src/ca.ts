import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime Web to find WASM files in node_modules
ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';

// Configuration
const MAP_SIZE = 72;
const PIXEL_SIZE = 8;
const CHANNEL_N = 16;
const ERASER_RADIUS = 3;

export class CellularAutomata {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private statusDiv: HTMLElement;
  private state: Float32Array | null = null;
  private session: ort.InferenceSession | null = null;
  private isRunning = false;
  private isMouseDown = false;
  private animationFrameId: number | null = null;
  private currentEmoji: number | null = null;

  constructor(canvas: HTMLCanvasElement, statusDiv: HTMLElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.statusDiv = statusDiv;
    this.setupMouseEvents();
  }

  // Initialize state with a single seed cell
  private initState(): void {
    this.state = new Float32Array(MAP_SIZE * MAP_SIZE * CHANNEL_N);
    const centerY = Math.floor(MAP_SIZE / 2);
    const centerX = Math.floor(MAP_SIZE / 2);
    const centerIdx = (centerY * MAP_SIZE + centerX) * CHANNEL_N;
    // Set alpha channel (channel 3) and all subsequent channels to 1.0
    for (let i = 3; i < CHANNEL_N; i++) {
      this.state[centerIdx + i] = 1.0;
    }
  }

  // Convert state to RGB for display
  private toRGB(stateData: Float32Array): ImageData {
    const imageData = this.ctx.createImageData(MAP_SIZE, MAP_SIZE);
    for (let y = 0; y < MAP_SIZE; y++) {
      for (let x = 0; x < MAP_SIZE; x++) {
        const stateIdx = (y * MAP_SIZE + x) * CHANNEL_N;
        const imageIdx = (y * MAP_SIZE + x) * 4;

        // Get RGB (premultiplied by alpha) and alpha
        const r = stateData[stateIdx + 0];
        const g = stateData[stateIdx + 1];
        const b = stateData[stateIdx + 2];
        const a = Math.max(0, Math.min(0.9999, stateData[stateIdx + 3]));

        // Convert to display RGB: 1.0 - a + rgb
        imageData.data[imageIdx + 0] = Math.floor(
          Math.min(255, Math.max(0, (1.0 - a + r) * 255))
        );
        imageData.data[imageIdx + 1] = Math.floor(
          Math.min(255, Math.max(0, (1.0 - a + g) * 255))
        );
        imageData.data[imageIdx + 2] = Math.floor(
          Math.min(255, Math.max(0, (1.0 - a + b) * 255))
        );
        imageData.data[imageIdx + 3] = 255;
      }
    }
    return imageData;
  }

  // Render the current state
  private render(): void {
    if (!this.state) return;

    const imageData = this.toRGB(this.state);
    // Create a temporary canvas to hold the small image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = MAP_SIZE;
    tempCanvas.height = MAP_SIZE;
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(imageData, 0, 0);

    // Scale up to the main canvas
    this.ctx.imageSmoothingEnabled = false; // Keep it pixelated
    this.ctx.drawImage(
      tempCanvas,
      0,
      0,
      MAP_SIZE,
      MAP_SIZE,
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );
  }

  // Run one step of the CA model
  private async step(): Promise<void> {
    if (!this.session || !this.state) return;

    try {
      // Reshape state for model input: [batch, height, width, channels]
      const inputTensor = new ort.Tensor('float32', this.state, [
        1,
        MAP_SIZE,
        MAP_SIZE,
        CHANNEL_N,
      ]);

      // Run inference
      const feeds = { x: inputTensor };
      const results = await this.session.run(feeds);
      const output = results.transpose_3;

      // Update state
      this.state = new Float32Array(output.data as Float32Array);

      this.render();
    } catch (error) {
      console.error('Error during inference:', error);
      this.statusDiv.textContent = 'Error during inference';
      this.isRunning = false;
    }
  }

  // Animation loop
  private async animate(): Promise<void> {
    if (this.isRunning) {
      await this.step();
    }
    this.animationFrameId = requestAnimationFrame(() => this.animate());
  }

  // Get grid position from mouse event
  private getGridPosition(event: MouseEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / PIXEL_SIZE);
    const y = Math.floor((event.clientY - rect.top) / PIXEL_SIZE);
    return { x, y };
  }

  // Erase cells at the given position
  private eraseAt(x: number, y: number): void {
    if (x < 0 || x >= MAP_SIZE || y < 0 || y >= MAP_SIZE || !this.state) return;

    for (let dy = -ERASER_RADIUS; dy <= ERASER_RADIUS; dy++) {
      for (let dx = -ERASER_RADIUS; dx <= ERASER_RADIUS; dx++) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist <= ERASER_RADIUS) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && nx < MAP_SIZE && ny >= 0 && ny < MAP_SIZE) {
            const idx = (ny * MAP_SIZE + nx) * CHANNEL_N;
            for (let c = 0; c < CHANNEL_N; c++) {
              this.state[idx + c] = 0;
            }
          }
        }
      }
    }
    this.render();
  }

  // Setup mouse event handlers
  private setupMouseEvents(): void {
    this.canvas.addEventListener('mousedown', (e) => {
      this.isMouseDown = true;
      const pos = this.getGridPosition(e);
      this.eraseAt(pos.x, pos.y);
    });

    this.canvas.addEventListener('mousemove', (e) => {
      if (this.isMouseDown) {
        const pos = this.getGridPosition(e);
        this.eraseAt(pos.x, pos.y);
      }
    });

    this.canvas.addEventListener('mouseup', () => {
      this.isMouseDown = false;
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.isMouseDown = false;
    });
  }

  // Public methods
  public async loadModel(emojiNum: number): Promise<void> {
    try {
      // Stop animation if running
      if (this.animationFrameId !== null) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
      }
      this.isRunning = false;

      this.statusDiv.textContent = `Loading emoji ${emojiNum} model...`;

      // Create ONNX session
      this.session = await ort.InferenceSession.create(
        `/models/emoji_${emojiNum}.onnx`,
        {
          executionProviders: ['wasm'],
        }
      );

      this.currentEmoji = emojiNum;
      this.statusDiv.textContent = `Emoji ${emojiNum} loaded! Running...`;

      // Initialize and render
      this.initState();
      this.render();

      // Start animation
      this.isRunning = true;
      this.animate();
    } catch (error) {
      console.error('Error loading model:', error);
      this.statusDiv.textContent = `Error loading emoji ${emojiNum}. Make sure models/emoji_${emojiNum}.onnx exists.`;
    }
  }

  public togglePause(): boolean {
    this.isRunning = !this.isRunning;
    return this.isRunning;
  }

  public reset(): void {
    this.initState();
    this.render();
  }

  public getCurrentEmoji(): number | null {
    return this.currentEmoji;
  }

  public clearCanvas(): void {
    this.ctx.fillStyle = '#f0f0f0';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
