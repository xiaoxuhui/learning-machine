/* Adapted from turbowarp-neural-network/src/network.mjs, MIT, xiaoxuhui. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.LearningModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const LIMIT = 1000, HIDDEN = 6, EPOCHS = 600, RATE = 0.08, MAX_SAMPLES = 32;

  function parseNumber(value) {
    if (typeof value !== 'string' && typeof value !== 'number') return null;
    if (String(value).trim() === '') return null;
    const n = Number(value);
    return Number.isFinite(n) && Math.abs(n) <= LIMIT ? n : null;
  }

  class NeuralNetwork {
    constructor() { this.reset(); }
    initialize() {
      this.w1 = Array.from({ length: HIDDEN }, (_, k) => (((k + 1) * 37 % 101) - 50) / 50);
      this.b1 = Array.from({ length: HIDDEN }, (_, k) => (((k + 1) * 19 % 31) - 15) / 100);
      this.w2 = Array.from({ length: HIDDEN }, (_, k) => (((k + 1) * 17 % 47) - 23) / 100);
      this.b2 = 0;
    }
    reset() {
      this.samples = []; this.epochs = 0; this.loss = null;
      this.sx = 1; this.sy = 1; this.initialize();
    }
    weights() { return { w1: [...this.w1], b1: [...this.b1], w2: [...this.w2], b2: this.b2 }; }
    activation(x) {
      return this.w1.map((w, j) => Math.tanh(Math.max(-10, Math.min(10, w * x + this.b1[j]))));
    }
    forward(x) { return this.b2 + this.activation(x).reduce((sum, v, j) => sum + v * this.w2[j], 0); }
    gradients(x, y) {
      const h = this.activation(x), d = 2 * (this.b2 + h.reduce((sum, v, j) => sum + v * this.w2[j], 0) - y);
      const dh = h.map((v, j) => d * this.w2[j] * (1 - v * v));
      return { w1: dh.map(v => v * x), b1: dh, w2: h.map(v => d * v), b2: d };
    }
    train(rawX, rawY) {
      const x = parseNumber(rawX), y = parseNumber(rawY);
      if (x === null || y === null) throw new RangeError('请输入 -1000 到 1000 之间的有效数字。');
      const index = this.samples.findIndex(s => s[0] === x);
      if (index < 0 && this.samples.length >= MAX_SAMPLES) throw new RangeError('最多学习 32 个不同输入；可以修改已有输入的正确输出。');
      if (index < 0) this.samples.push([x, y]); else this.samples[index] = [x, y];
      this.sx = Math.max(1, ...this.samples.map(s => Math.abs(s[0])));
      this.sy = Math.max(1, ...this.samples.map(s => Math.abs(s[1])));
      this.initialize();
      for (let epoch = 0; epoch < EPOCHS; epoch++) {
        for (const [vx, vy] of this.samples) {
          const g = this.gradients(vx / this.sx, vy / this.sy);
          for (let j = 0; j < HIDDEN; j++) {
            this.w1[j] -= RATE * g.w1[j]; this.b1[j] -= RATE * g.b1[j]; this.w2[j] -= RATE * g.w2[j];
          }
          this.b2 -= RATE * g.b2;
        }
      }
      this.epochs += EPOCHS;
      this.loss = this.samples.reduce((sum, [vx, vy]) => sum + (this.forward(vx / this.sx) * this.sy - vy) ** 2, 0) / this.samples.length;
      return this.loss;
    }
    predict(value) {
      if (!this.samples.length) throw new Error('先教我至少一组输入和正确输出，再来计算吧。');
      const x = parseNumber(value);
      if (x === null) throw new RangeError('请输入 -1000 到 1000 之间的有效数字。');
      return this.forward(x / this.sx) * this.sy;
    }
  }
  return { NeuralNetwork, parseNumber, LIMIT, HIDDEN, EPOCHS, MAX_SAMPLES };
});
