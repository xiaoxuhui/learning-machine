/* Extends the MIT turbowarp-neural-network gradient-descent model to real multilayer networks. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LearningModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const LIMIT = 1000, EPOCHS = 600, RATE = 0.08, MAX_SAMPLES = 32, TARGET_LOSS = 0.0001;
  const CANDIDATES = [[2], [4], [6], [4, 3], [6, 4], [8, 6, 4]];
  function parseNumber(value) {
    if (typeof value !== 'string' && typeof value !== 'number') return null;
    if (String(value).trim() === '') return null;
    const n = Number(value);
    return Number.isFinite(n) && Math.abs(n) <= LIMIT ? n : null;
  }
  const copy = value => JSON.parse(JSON.stringify(value));
  class NeuralNetwork {
    constructor() { this.reset(); }
    initialize(architecture = [2]) {
      this.architecture = [...architecture];
      let seed = 109;
      const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
      const sizes = [1, ...architecture, 1];
      this.layers = sizes.slice(1).map((size, i) => {
        const scale = Math.sqrt(6 / (sizes[i] + size));
        return { weights: Array.from({ length: size }, () => Array.from({ length: sizes[i] }, () => (random() * 2 - 1) * scale)), biases: Array.from({ length: size }, () => (random() - 0.5) * 0.1) };
      });
    }
    reset() {
      this.samples = []; this.epochs = 0; this.loss = null; this.sx = 1; this.sy = 1;
      this.structureMode = 'auto'; this.trials = []; this.lastRounds = 0;
      this.initialize();
    }
    weights() { return copy(this.layers); }
    traceNormalized(x) {
      let inputs = [x];
      return this.layers.map((layer, index) => {
        const z = layer.weights.map((row, i) => row.reduce((sum, w, j) => sum + w * inputs[j], layer.biases[i]));
        const outputs = index === this.layers.length - 1 ? z : z.map(Math.tanh);
        const trace = { inputs, z, outputs }; inputs = outputs; return trace;
      });
    }
    forward(x) { return this.traceNormalized(x).at(-1).outputs[0]; }
    gradients(x, y) {
      const trace = this.traceNormalized(x), gradients = new Array(this.layers.length);
      let delta = [2 * (trace.at(-1).outputs[0] - y)];
      for (let l = this.layers.length - 1; l >= 0; l--) {
        gradients[l] = { weights: delta.map(d => trace[l].inputs.map(v => d * v)), biases: [...delta] };
        if (l > 0) {
          delta = trace[l - 1].outputs.map((v, i) => delta.reduce((sum, d, j) => sum + this.layers[l].weights[j][i] * d, 0) * (1 - v * v));
        }
      }
      return gradients;
    }
    fit() {
      const candidates = this.structureMode === 'auto' ? CANDIDATES : [this.structureMode.split('-').map(Number)];
      let best = null; this.trials = []; this.lastRounds = 0;
      for (const architecture of candidates) {
        this.initialize(architecture);
        for (let epoch = 0; epoch < EPOCHS; epoch++) {
          for (const [x, y] of this.samples) {
            const g = this.gradients(x / this.sx, y / this.sy);
            this.layers.forEach((layer, l) => {
              layer.weights.forEach((row, i) => {
                row.forEach((_, j) => { row[j] -= RATE * g[l].weights[i][j]; });
                layer.biases[i] -= RATE * g[l].biases[i];
              });
            });
          }
        }
        this.lastRounds += EPOCHS;
        const normalizedLoss = this.samples.reduce((sum, [x, y]) => sum + (this.forward(x / this.sx) - y / this.sy) ** 2, 0) / this.samples.length;
        this.trials.push({ architecture: [...architecture], normalizedLoss });
        if (!best || normalizedLoss < best.normalizedLoss) best = { architecture: [...architecture], layers: this.weights(), normalizedLoss };
        if (normalizedLoss <= TARGET_LOSS) break;
      }
      this.architecture = best.architecture; this.layers = best.layers;
      this.loss = best.normalizedLoss * this.sy ** 2; this.epochs += this.lastRounds;
    }
    train(rawX, rawY) {
      const x = parseNumber(rawX), y = parseNumber(rawY);
      if (x === null || y === null) throw new RangeError('请输入 -1000 到 1000 之间的有效数字。');
      const index = this.samples.findIndex(s => s[0] === x);
      if (index < 0 && this.samples.length >= MAX_SAMPLES) throw new RangeError('最多学习 32 个不同输入；可以修改已有输入的正确输出。');
      if (index < 0) this.samples.push([x, y]); else this.samples[index] = [x, y];
      this.sx = Math.max(1, ...this.samples.map(s => Math.abs(s[0])));
      this.sy = Math.max(1, ...this.samples.map(s => Math.abs(s[1])));
      this.fit(); return this.loss;
    }
    setStructure(mode) {
      if (mode !== 'auto' && !CANDIDATES.some(a => a.join('-') === mode)) throw new RangeError('不支持的网络结构。');
      this.structureMode = mode;
      if (this.samples.length) this.fit();
      else { this.initialize(mode === 'auto' ? [2] : mode.split('-').map(Number)); this.trials = []; this.lastRounds = 0; }
    }
    predict(value) {
      if (!this.samples.length) throw new Error('先教我至少一组输入和正确输出，再来计算吧。');
      const x = parseNumber(value);
      if (x === null) throw new RangeError('请输入 -1000 到 1000 之间的有效数字。');
      return this.forward(x / this.sx) * this.sy;
    }
    forwardTrace(value) {
      const x = parseNumber(value);
      if (x === null) throw new RangeError('无效输入。');
      const normalizedX = x / this.sx, layers = this.traceNormalized(normalizedX);
      return { x, normalizedX, layers, output: layers.at(-1).outputs[0] * this.sy };
    }
    functionExpression() {
      let inputs = ['(x / ' + this.sx + ')'];
      this.layers.forEach((layer, l) => {
        inputs = layer.weights.map((row, i) => {
          const inner = '(' + layer.biases[i] + ' + ' + row.map((w, j) => '(' + w + ') * ' + inputs[j]).join(' + ') + ')';
          return l === this.layers.length - 1 ? inner : 'tanh' + inner;
        });
      });
      return this.sy + ' * ' + inputs[0];
    }
    equations() {
      const n = value => String(Number(value.toPrecision(6)));
      let inputs = ['u'];
      const equations = [{ name: 'u', expression: 'x / ' + n(this.sx), layer: 0 }];
      this.layers.forEach((layer, l) => {
        const names = layer.weights.map((row, i) => {
          const name = l === this.layers.length - 1 ? 'v' : 'h' + (l + 1) + '_' + (i + 1);
          const sum = n(layer.biases[i]) + ' + ' + row.map((w, j) => '(' + n(w) + ') × ' + inputs[j]).join(' + ');
          equations.push({ name, expression: l === this.layers.length - 1 ? sum : 'tanh(' + sum + ')', layer: l + 1 });
          return name;
        }); inputs = names;
      });
      equations.push({ name: 'f(x)', expression: n(this.sy) + ' × v', layer: this.layers.length + 1 });
      return equations;
    }
  }
  return { NeuralNetwork, parseNumber, LIMIT, HIDDEN: 2, EPOCHS, MAX_SAMPLES, TARGET_LOSS, CANDIDATES };
});
