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
  assert.equal(m.epochs, 4200); assert.ok(Number.isFinite(m.loss)); assert.ok(m.loss < 0.001);
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
  assert.deepEqual(m.samples, [[1, -3]]); assert.equal(m.epochs, 8400);
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
  assert.ok(m.architecture.length > 1); assert.equal(m.trials.length, 7);
  assert.equal(m.lastRounds, 4200);
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

test('four custom hidden layers and multiple endpoints have correct real matrix shapes', () => {
  const m = new NeuralNetwork({ inputSize: 3, outputSize: 2 }); m.setStructure('3-5-4-2');
  assert.deepEqual(m.layers.map(l => [l.weights.length, l.weights[0].length]), [[3, 3], [5, 3], [4, 5], [2, 4], [2, 2]]);
  m.train([1, -2, 0.5], [3, -4]);
  assert.ok(m.layers.every(l => l.weights.flat().every(Number.isFinite)));
  assert.equal(m.forwardTrace([1, -2, 0.5]).layers.length, 5);
});
test('learns two-input sum and difference with independent two-output values', () => {
  const m = new NeuralNetwork({ inputSize: 2, outputSize: 2 });
  for (const pair of [[[0, 0], [0, 0]], [[1, 0], [1, 1]], [[0, 1], [1, -1]], [[1, 1], [2, 0]]]) m.train(...pair);
  const before = JSON.stringify(m), result = m.predict([0.5, 0.25]);
  assert.ok(Math.abs(result[0] - 0.75) < 0.15); assert.ok(Math.abs(result[1] - 0.25) < 0.15);
  assert.equal(JSON.stringify(m), before);
});
test('multi-output loss gradient agrees with independently averaged finite difference loss', () => {
  const m = new NeuralNetwork({ inputSize: 2, outputSize: 3 }); m.setStructure('3-2-2-2');
  const x = [0.4, -0.2], y = [0.3, -0.1, 0.8], g = m.gradients(x, y), epsilon = 1e-5;
  const loss = () => m.forward(x).reduce((sum, v, i) => sum + (v - y[i]) ** 2, 0) / 3;
  for (let l = 0; l < m.layers.length; l++) {
    for (let i = 0; i < m.layers[l].weights.length; i++) {
      for (let j = 0; j < m.layers[l].weights[i].length; j++) {
        const original = m.layers[l].weights[i][j];
        m.layers[l].weights[i][j] = original + epsilon; const plus = loss();
        m.layers[l].weights[i][j] = original - epsilon; const minus = loss(); m.layers[l].weights[i][j] = original;
        assert.ok(Math.abs((plus - minus) / (2 * epsilon) - g[l].weights[i][j]) < 1e-6);
      }
      const original = m.layers[l].biases[i];
      m.layers[l].biases[i] = original + epsilon; const plus = loss();
      m.layers[l].biases[i] = original - epsilon; const minus = loss(); m.layers[l].biases[i] = original;
      assert.ok(Math.abs((plus - minus) / (2 * epsilon) - g[l].biases[i]) < 1e-6);
    }
  }
});
test('each vector output function matches real inference, including bounded large expansions', () => {
  for (const structure of ['4-3', '3-3-3-3', '12-12-12-12']) {
    const m = new NeuralNetwork({ inputSize: 2, outputSize: 2 }); m.setStructure(structure); m.train([-2, 0.5], [2, -3]);
    for (let i = 0; i < 2; i++) {
      const expression = m.functionExpression(i); assert.ok(expression.length < 60000);
      const calculate = new Function('x1', 'x2', 'tanh', `return ${expression};`);
      for (const x of [[-1, 0.25], [0, 0], [1000, -1000]]) assert.ok(Math.abs(calculate(...x, Math.tanh) - m.predict(x)[i]) < 1e-10);
    }
  }
});
test('duplicate vectors match every component; malformed dimensions are rejected atomically', () => {
  const m = new NeuralNetwork({ inputSize: 2, outputSize: 2 });
  m.train([1, 2], [3, 4]); m.train([1, 3], [4, 5]); m.train([1, 2], [-1, -2]);
  assert.equal(m.samples.length, 2); assert.deepEqual(m.samples[0], [[1, 2], [-1, -2]]);
  const before = JSON.stringify(m);
  for (const pair of [[[1], [2, 3]], [[1, 2], [3]], [[1, ''], [0, 0]], [[1, 2], [0, Infinity]]]) {
    assert.throws(() => m.train(...pair), RangeError); assert.equal(JSON.stringify(m), before);
  }
});
test('dimension changes clear incompatible data; same dimensions preserve samples; reset restores defaults', () => {
  const m = new NeuralNetwork(); const initial = JSON.stringify(m); m.train(1, 2);
  m.configure({ structure: '2-3-4-2' }); assert.deepEqual(m.samples, [[1, 2]]);
  assert.deepEqual(m.configure({ inputSize: 2, outputSize: 3 }), { dimensionsChanged: true });
  assert.equal(m.samples.length, 0); assert.equal(m.epochs, 0); assert.equal(m.loss, null);
  assert.deepEqual(m.inputScales, [1, 1]); assert.deepEqual(m.outputScales, [1, 1, 1]);
  m.reset(); assert.equal(JSON.stringify(m), initial);
});
test('configuration errors never change an existing trained model', () => {
  const m = new NeuralNetwork(); m.train(1, 2); const before = JSON.stringify(m);
  for (const options of [{ inputSize: 0 }, { outputSize: 5 }, { inputSize: 1.5 }, { structure: '2-2-2-2-2' }, { structure: '0' }, { structure: '13' }]) {
    assert.throws(() => m.configure(options), RangeError); assert.equal(JSON.stringify(m), before);
  }
});

test('automatic training evaluates every width and depth even when the first candidate fits', () => {
  const m = new NeuralNetwork(); m.train(1, 2);
  assert.deepEqual(m.trials.map(t => t.architecture), [[2], [4], [6], [4, 3], [6, 4], [8, 6, 4], [8, 6, 4, 2]]);
  assert.ok(m.trials.every(t => Number.isFinite(t.normalizedLoss) && t.parameterCount > 0));
  assert.equal(m.trials.filter(t => t.selected).length, 1);
  assert.equal(m.decision.tested, 7); assert.equal(m.decision.reason, 'simplest-meeting-target');
  const eligible = m.trials.filter(t => t.normalizedLoss <= 0.0001);
  assert.equal(m.decision.parameterCount, Math.min(...eligible.map(t => t.parameterCount)));
});
test('without choosing any structure, examples cause automatic deepening and simpler corrected examples cause shallowing', () => {
  const m = new NeuralNetwork();
  for (const pair of [[-2, 0], [-1, 1], [0, 0], [1, 1], [2, 0]]) m.train(...pair);
  assert.equal(m.structureMode, 'auto'); assert.ok(m.architecture.length > 1);
  assert.ok(m.shapeHistory.some(d => d.action === 'deepen' && d.mode === 'auto'));
  for (const x of [-2, -1, 0, 1, 2]) m.train(x, 0);
  assert.deepEqual(m.architecture, [2]); assert.ok(m.shapeHistory.some(d => d.action === 'shallow'));
  assert.equal(m.samples.length, 5);
});
test('explicitly fixed structure can return to autonomous decisions without losing samples', () => {
  const m = new NeuralNetwork(); m.setStructure('4-3'); m.train(1, 2);
  assert.equal(m.trials.length, 1); assert.equal(m.decision.reason, 'fixed-by-user');
  const samples = JSON.stringify(m.samples); m.setStructure('auto');
  assert.equal(JSON.stringify(m.samples), samples); assert.equal(m.decision.mode, 'auto'); assert.equal(m.trials.length, 7);
  assert.deepEqual(m.architecture, [2]);
  const before = JSON.stringify(m); m.predict(0.5); m.forwardTrace(0.2); m.functionExpression(); assert.equal(JSON.stringify(m), before);
});
test('decision history is bounded and reset clears every decision', () => {
  const m = new NeuralNetwork(); for (let i = 0; i < 12; i++) m.train(0, i);
  assert.equal(m.shapeHistory.length, 10); assert.deepEqual(m.shapeHistory.at(-1), m.decision);
  m.reset(); assert.equal(m.decision, null); assert.deepEqual(m.shapeHistory, []); assert.deepEqual(m.trials, []);
});
