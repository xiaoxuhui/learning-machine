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
