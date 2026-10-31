const { test } = require('node:test');
const assert = require('node:assert/strict');
const { NeuralNetwork, parseNumber } = require('../src/model.js');

test('initial state and reset restore all original parameters', () => {
  const model = new NeuralNetwork(), initial = JSON.stringify(model);
  model.train(2, 4); model.predict(3); model.reset();
  assert.equal(JSON.stringify(model), initial);
});
test('training changes real weights and records finite loss', () => {
  const m = new NeuralNetwork(), weights = m.weights(); m.train(1, 2);
  assert.notDeepEqual(m.weights(), weights); assert.equal(m.samples.length, 1);
  assert.equal(m.epochs, 600); assert.ok(Number.isFinite(m.loss)); assert.ok(m.loss < 0.001);
});
test('learns an unseen input for the double rule', () => {
  const m = new NeuralNetwork(); for (const pair of [[0, 0], [1, 2], [2, 4]]) m.train(...pair);
  assert.ok(Math.abs(m.predict(1.5) - 3) < 0.35);
});
test('prediction does not mutate any model state', () => {
  const m = new NeuralNetwork(); m.train(1, 2); const before = JSON.stringify(m);
  m.predict(-0.5); m.predict(1000); assert.equal(JSON.stringify(m), before);
});
test('duplicate input updates the correct output', () => {
  const m = new NeuralNetwork(); m.train(1, 2); m.train(1, -3);
  assert.deepEqual(m.samples, [[1, -3]]); assert.equal(m.epochs, 1200);
  assert.ok(Math.abs(m.predict(1) + 3) < 0.01);
});
test('empty, non-finite, invalid and out of range numbers are rejected', () => {
  for (const input of ['', ' ', 'abc', '1,2', NaN, Infinity, '-Infinity', 1001, -1001, null, undefined, true, [], {}]) assert.equal(parseNumber(input), null);
});
test('zero, decimals, negatives and inclusive bounds are valid', () => {
  for (const input of ['0', ' -.5 ', '1e2', '-1000', 1000]) assert.equal(parseNumber(input), Number(input));
});
test('invalid training leaves an already trained model unchanged', () => {
  const m = new NeuralNetwork(); m.train(1, 2); const before = JSON.stringify(m);
  for (const pair of [['', 2], [2, ''], [Infinity, 0], [0, 1001]]) {
    assert.throws(() => m.train(...pair), RangeError); assert.equal(JSON.stringify(m), before);
  }
});
test('32 samples allowed, 33rd rejected, existing input still editable', () => {
  const m = new NeuralNetwork(); for (let i = 0; i < 32; i++) m.train(i, i * 2);
  const before = JSON.stringify(m); assert.throws(() => m.train(32, 64), /32/);
  assert.equal(JSON.stringify(m), before); m.train(0, 3);
  assert.equal(m.samples.length, 32); assert.deepEqual(m.samples[0], [0, 3]);
});
test('untrained prediction gives an actionable message', () => {
  assert.throws(() => new NeuralNetwork().predict(0), /先教我/);
});
test('extreme samples keep all weights and predictions finite', () => {
  const m = new NeuralNetwork(); for (const pair of [[-1000, 1000], [0, 0], [1000, -1000]]) m.train(...pair);
  assert.ok(Number.isFinite(m.loss)); for (const x of [-1000, 0, 1000]) assert.ok(Number.isFinite(m.predict(x)));
});
test('same examples give reproducible predictions', () => {
  const a = new NeuralNetwork(), b = new NeuralNetwork();
  for (const pair of [[-2, 4], [0, 0], [2, 4]]) { a.train(...pair); b.train(...pair); }
  assert.deepEqual(a.weights(), b.weights());
});

test('three hidden layers have real matching matrix dimensions', () => {
  const m = new NeuralNetwork(); m.setStructure('8-6-4');
  assert.deepEqual(m.architecture, [8, 6, 4]);
  assert.deepEqual(m.layers.map(l => [l.weights.length, l.weights[0].length]), [[8, 1], [6, 8], [4, 6], [1, 4]]);
  const before = m.weights(); m.train(0.5, -1);
  m.layers.forEach((l, i) => assert.notDeepEqual(l.weights, before[i].weights));
  assert.ok(Math.abs(m.predict(0.5) + 1) < 0.05);
});
test('deep network analytic gradients match independent finite differences', () => {
  const m = new NeuralNetwork(); m.setStructure('4-3');
  const x = 0.37, y = -0.21, g = m.gradients(x, y), epsilon = 1e-5;
  const loss = () => (m.forward(x) - y) ** 2;
  for (let l = 0; l < m.layers.length; l++) {
    const layer = m.layers[l];
    for (let i = 0; i < layer.weights.length; i++) {
      for (let j = 0; j < layer.weights[i].length; j++) {
        const original = layer.weights[i][j];
        layer.weights[i][j] = original + epsilon; const plus = loss();
        layer.weights[i][j] = original - epsilon; const minus = loss();
        layer.weights[i][j] = original;
        assert.ok(Math.abs((plus - minus) / (2 * epsilon) - g[l].weights[i][j]) < 1e-6, `weight ${l}/${i}/${j}`);
      }
      const original = layer.biases[i];
      layer.biases[i] = original + epsilon; const plus = loss();
      layer.biases[i] = original - epsilon; const minus = loss(); layer.biases[i] = original;
      assert.ok(Math.abs((plus - minus) / (2 * epsilon) - g[l].biases[i]) < 1e-6, `bias ${l}/${i}`);
    }
  }
});
test('full nested expression and layer traces match actual inference for every supported structure', () => {
  for (const mode of ['2', '4', '6', '4-3', '6-4', '8-6-4']) {
    const m = new NeuralNetwork(); m.setStructure(mode); m.train(-2, 3); m.train(2, -1);
    const expression = new Function('x', 'tanh', `return ${m.functionExpression()};`);
    for (const x of [-1000, -0.75, 0, 1.5, 1000]) {
      assert.ok(Math.abs(expression(x, Math.tanh) - m.predict(x)) < 1e-10);
      const trace = m.forwardTrace(x);
      assert.equal(trace.output, m.predict(x));
      assert.equal(trace.layers.length, m.architecture.length + 1);
      for (let l = 1; l < trace.layers.length; l++) assert.deepEqual(trace.layers[l].inputs, trace.layers[l - 1].outputs);
    }
  }
});
test('auto selection really changes width and may select nested layers based on observed error', () => {
  const m = new NeuralNetwork();
  for (const p of [[-2, 0], [-1, 1], [0, 0], [1, 1], [2, 0]]) m.train(...p);
  assert.ok(m.architecture.length > 1); assert.equal(m.trials.length, 6);
  assert.equal(m.lastRounds, 3600);
  const minimum = Math.min(...m.trials.map(t => t.normalizedLoss));
  assert.ok(Math.abs(m.loss / m.sy ** 2 - minimum) < 1e-12);
});
test('reset after nested learning clears structure, trials, rounds, samples and all parameters', () => {
  const m = new NeuralNetwork(), initial = JSON.stringify(m);
  m.setStructure('8-6-4'); m.train(1, -2); m.reset();
  assert.equal(JSON.stringify(m), initial);
});
test('structure choice retrains existing samples and never fabricates extra examples', () => {
  const m = new NeuralNetwork(); m.train(1, 2); const samples = JSON.stringify(m.samples), epochs = m.epochs;
  m.setStructure('4-3'); assert.equal(JSON.stringify(m.samples), samples);
  assert.equal(m.epochs, epochs + 600); assert.deepEqual(m.architecture, [4, 3]);
  assert.ok(Math.abs(m.predict(1) - 2) < 0.02);
  const before = JSON.stringify(m); assert.throws(() => m.setStructure('999'), RangeError); assert.equal(JSON.stringify(m), before);
});
