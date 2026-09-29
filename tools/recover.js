const fs = require('fs');
const p = 'src/main/resources/index.html';
let t = fs.readFileSync(p, 'utf8');
// Reverse the UTF-8 -> cp1252 mis-decode: map every char back to its source byte.
const bytes = [];
for (const ch of t) {
  const cp = ch.codePointAt(0);
  if (cp < 256) { bytes.push(cp); continue; }
  const enc = Buffer.from(ch, 'utf8');
  // cp1252-encodable chars (dagger, euro, quotes, dashes, ...) -> single byte
  try {
    const b = Buffer.from(ch, 'binary');
    const back = Buffer.from(b.toString('latin1'), 'utf8');
    const single = Buffer.alloc(1);
    // use TextEncoder-free manual map via iconv-lite-free approach:
    const map = {'\u2020':0x86,'\u2019':0x92,'\u201c':0x93,'\u201d':0x94,'\u2013':0x96,
      '\u2014':0x97,'\u02dc':0x98,'\u02c6':0x88,'\u0153':0x9b,'\u0152':0x8c,
      '\u201a':0x82,'\u0192':0x83,'\u201e':0x84,'\u2026':0x85,'\u2021':0x87,
      '\u0160':0x8a,'\u2039':0x8b,'\u0161':0x8c+1,'\u203a':0x8b+1,'\u0152':0x8c,
      '\u0178':0x9f,'\u0161':0x9a,'\u2030':0x89,'\u20ac':0x80,'\u2018':0x91,
      '\u00a0':0xa0};
    void enc; void back; void b; void single;
    if (map[ch] !== undefined) { bytes.push(map[ch]); continue; }
    throw new Error('no map for ' + ch + ' U+' + cp.toString(16));
  } catch (e) { throw new Error('at char U+' + cp.toString(16) + ': ' + e.message); }
}
let f = Buffer.from(bytes).toString('utf8');
fs.writeFileSync(p + '.fixed', f, 'utf8');
const rest = [...new Set([...f].filter(c => c.codePointAt(0) > 127))].sort();
console.log('recovered chars:', f.length);
console.log('remaining non-ascii:', rest.length);
console.log(rest.map(c => 'U+' + c.codePointAt(0).toString(16) + ' ' + c).join('\n'));
