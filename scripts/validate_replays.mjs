// Validate the actual published dataset without initializing a neural model.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {createPackedFetch} from '../viewer/transport.mjs';
import {createLoader, Replay, checkedBytes} from '../viewer/replay.mjs';
import {decodeAdjacency} from '../viewer/connections.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const cache = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, '.cache/replay');
const raw = async name => new Response(await fs.readFile(name.startsWith('data/packed/')
  ? path.join(cache, path.basename(name)) : path.join(root, 'viewer', name)));
const fetcher = createPackedFetch(raw);
const manifest = await (await fetcher('data/full-selected/manifest.json')).json();
const anatomy = JSON.parse(new TextDecoder().decode(await checkedBytes(
  await fetcher(manifest.anatomy_file), manifest.anatomy_sha256)));
assert.equal(anatomy.nodes.length, 165122);
assert.equal(new Set(anatomy.nodes.map(n => n.id)).size, 165122);
anatomy.nodes.forEach((n, i) => {
  assert.equal(typeof n.id, 'string');
  assert.equal(n.original_index, i);
  if (n.position) assert(n.position.length === 3 && n.position.every(Number.isFinite));
});
assert.equal(anatomy.nodes.filter(n => n.position).length, 141000);
await checkedBytes(await fetcher(manifest.positions_file), manifest.positions_sha256);
assert.equal(anatomy.selected_efficacy.checkpoint_sha256, manifest.checkpoint_sha256);

// Match original wall/tail/growth rules. New food is recorded; its placement is
// checked for validity, but Python's seeded RNG is not reimplemented here.
function checkMoves(record) {
  let before = record.initial, sinceFood = 0;
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  for (const frame of record.frames) {
    assert.deepEqual(frame.before, before);
    assert(!before.done);
    const d = (before.direction + frame.action + 3) % 4;
    const [dx, dy] = [[0,-1],[1,0],[0,1],[-1,0]][d];
    const h = [before.body[0][0] + dx, before.body[0][1] + dy];
    const eating = same(h, before.food);
    const occupied = eating ? before.body : before.body.slice(0, -1);
    const collision = h.some(x => x < 0 || x >= before.size) || occupied.some(p => same(p, h));
    const after = frame.after;
    assert.equal(after.moves, before.moves + 1);
    assert.equal(after.direction, d);
    sinceFood++;
    if (collision) {
      assert.deepEqual(after.body, before.body);
      assert.equal(after.score, before.score);
      assert(after.done && after.reason === 'Collision');
    } else {
      const body = [h, ...before.body];
      if (!eating) body.pop();
      else sinceFood = 0;
      assert.deepEqual(after.body, body);
      assert.equal(after.score, before.score + Number(eating));
      if (!eating) assert.deepEqual(after.food, before.food);
      else if (after.food) {
        assert(after.food.every(x => Number.isInteger(x) && x >= 0 && x < before.size));
        assert(!body.some(p => same(p, after.food)));
      }
      assert.equal(after.done, body.length === before.size ** 2 || sinceFood > 4 * before.size ** 2);
    }
    // Exact render values of the pre-action observation.
    const pixels = new Float32Array(before.size ** 2);
    for (const [x, y] of before.body) pixels[y * before.size + x] = .5;
    pixels[before.body[0][1] * before.size + before.body[0][0]] = .8;
    if (before.food) pixels[before.food[1] * before.size + before.food[0]] = 1;
    assert.deepEqual(Float32Array.from(frame.image), pixels);
    before = after;
  }
}

for (const entry of manifest.catalog) {
  let loaded;
  assert(await createLoader(manifest, value => loaded = value, fetcher)(entry));
  checkMoves(loaded.record);
  const replay = new Replay(loaded.record, loaded.activity, loaded.substeps);
  for (let i = 0; i <= entry.moves; i++) {
    replay.seek(i);
    assert.equal(replay.board.moves, i);
    assert.equal(replay.time, i * 100);
    assert.equal(replay.values.length, 165122);
  }
  assert(replay.board.done);
  assert.equal(replay.board.score, entry.score);
  console.log(`${entry.id}: ${entry.moves} decisions, ${entry.score} food; hashes, images and transitions verified`);
}

for (const direction of ['incoming', 'outgoing']) {
  let pairs = 0, synapses = 0, next = 0;
  for (const entry of anatomy.connections[direction]) {
    const bytes = await checkedBytes(await fetcher(entry.file), entry.sha256);
    const efficacy = new Float32Array(await checkedBytes(await fetcher(entry.efficacy_file), entry.efficacy_sha256));
    const u = new Uint32Array(bytes);
    const [version, start, count, edges] = u, header = 5 + count;
    assert.equal(version, 1); assert.equal(start, next);
    assert.equal(u.length, header + 3 * edges); assert.equal(efficacy.length, edges);
    assert.equal(u[4], 0); assert.equal(u[4 + count], edges);
    for (let j = 0; j < count; j++) assert(u[4+j] <= u[5+j]);
    for (let j = 0; j < edges; j++) {
      assert(u[header+3*j] < 165122 && u[header+3*j+1] > 0);
      assert(Number.isFinite(efficacy[j]) && efficacy[j] >= .049999 && efficacy[j] <= 4.000001);
      synapses += u[header+3*j+1];
    }
    decodeAdjacency(bytes, start, direction, 165122, efficacy);
    pairs += edges; next += count;
  }
  assert.equal(next, 165122); assert.equal(pairs, 25563197); assert.equal(synapses, 124025046);
  console.log(`${direction}: all ${pairs} directed rows verified`);
}
