// Structural validation of the PDF writer inside src/main/resources/index.html.
// Extracts the browser PDF helpers, builds a report PDF with text + one 1x1 JPEG figure and
// verifies the header, the xref offsets, the page tree and the trailer.
const fs = require('fs');
const html = fs.readFileSync('src/main/resources/index.html', 'utf8');
const start = html.indexOf('function pdfAscii');
const end = html.indexOf('function reportFigures');
if (start < 0 || end < 0) { console.error('PDF helpers not found in index.html'); process.exit(1); }
const helpers = html.substring(start, end);
const sandbox = {atob: s => Buffer.from(s, 'base64').toString('binary')};
const factory = new Function('atob', helpers + '\nreturn {buildPdfBlob,layoutPdf,wrapPdf,escapePdf,base64Bytes,pdfBytes};');
const pdf = factory(sandbox.atob);

const items = [
  {text: 'DyMETER run report', style: 'title'},
  {text: 'Title: DyMETER streaming anomaly detection report - sample_stream.csv', style: 'body'},
  {text: 'Generated: 2026-09-29 12:00:00', style: 'body'},
  {text: '1. Dataset', style: 'h2'},
  {text: 'File: sample_stream.csv    Data rows: 1500    Feature columns: 8', style: 'body'},
  {text: '2. Concept drift events (row numbers and reasons)', style: 'h2'},
  {text: 'Row 82 - concept 2 (U_ctr = 0.0321, R_e = 0.1140, mu* at onset = 0.4011, flagged: normal)', style: 'body'},
  {text: 'Reason: Concept 2 declared at stream row 82: the smoothed concept uncertainty stayed above mu_e ' +
         'for 5 consecutive instances (Algorithm 2, line 3) - parentheses ( ) and backslash \\ must be escaped.', style: 'body'}
];
for (let i = 0; i < 120; i++) items.push({text: 'Filler line ' + i + ' to force pagination across several pages.', style: 'body'});

const tinyJpegBase64 = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a' +
  'HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/E' +
  'ABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q==';
const image = {name:'chart1', index:0, caption:'Figure A - anomaly score versus threshold',
  width:1, height:1, data:pdf.base64Bytes('data:image/jpeg;base64,' + tinyJpegBase64)};

const pages = pdf.layoutPdf(items, [image]);
const blob = pdf.buildPdfBlob(pages, [image]);
blob.arrayBuffer().then(buffer => {
  const bytes = Buffer.from(buffer);
  const text = bytes.toString('latin1');
  const checks = [];
  checks.push(['header %PDF-1.4', text.startsWith('%PDF-1.4')]);
  checks.push(['binary marker', bytes[9] === 0x25 && bytes[10] === 0xE2]);
  checks.push(['ends with %%EOF', text.trimEnd().endsWith('%%EOF')]);
  checks.push(['more than one page', pages.length > 3]);
  const kids = /\/Kids \[([^\]]*)\]/.exec(text);
  checks.push(['page tree lists every page', kids && kids[1].trim().split(/\s+/).length === pages.length * 3]);
  checks.push(['JPEG image embedded', /\/Filter \/DCTDecode/.test(text) && /\/Im0 Do/.test(text)]);
  checks.push(['Times fonts used', /\/BaseFont \/Times-Roman/.test(text) && /\/BaseFont \/Times-Bold/.test(text)]);
  checks.push(['parentheses escaped', /\\\( \)\\ /.test(text) || text.includes('\\(')]);
  // xref offsets must point exactly at "N 0 obj"
  const xref = /xref\n0 (\d+)\n([\s\S]*?)trailer/.exec(text);
  const count = xref ? parseInt(xref[1], 10) : 0;
  if (!xref) console.error('xref block not found; tail = ' + JSON.stringify(text.slice(-200)));
  const entries = xref ? xref[2].trimEnd().split('\n') : [];
  let offsetsOk = entries.length === count;
  for (let i = 1; i < entries.length; i++) {
    const offset = parseInt(entries[i].substring(0, 10), 10);
    const header = text.substring(offset, offset + 12);
    if (!header.startsWith(i + ' 0 obj')) { offsetsOk = false; console.error('bad offset for object ' + i + ': ' + JSON.stringify(header)); }
  }
  checks.push(['every xref offset points at its object', offsetsOk]);
  checks.push(['startxref resolves to the xref table', (() => {
    const marker = text.lastIndexOf('startxref');
    const value = parseInt(text.substring(marker + 9).trim().split(/\s+/)[0], 10);
    return text.substring(value, value + 4) === 'xref';
  })()]);
  fs.writeFileSync('docs/results/pdf-structure-test.pdf', bytes);
  let failed = 0;
  checks.forEach(([name, ok]) => { if (!ok) failed++; console.log((ok ? '  [ok]   ' : '  [FAIL] ') + name); });
  console.log('pages: ' + pages.length + ', bytes: ' + bytes.length + ', objects: ' + count);
  console.log(failed === 0 ? 'PDF structure OK' : failed + ' checks failed');
  process.exit(failed === 0 ? 0 : 1);
});
