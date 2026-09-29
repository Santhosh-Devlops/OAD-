// Lists the non-ASCII sequences left in the dashboard so they can be normalised to ASCII.
const fs = require('fs');
const text = fs.readFileSync('src/main/resources/index.html', 'utf8');
const counts = new Map();
for (const character of text) {
  const code = character.codePointAt(0);
  if (code > 126) counts.set(character, (counts.get(character) || 0) + 1);
}
console.log('distinct non-ASCII characters:', counts.size);
[...counts.entries()].sort((a, b) => b[1] - a[1]).forEach(([character, count]) => {
  console.log('  ' + JSON.stringify(character) + '  U+' + character.codePointAt(0).toString(16).toUpperCase() +
    '  x' + count + '   context: ' + JSON.stringify(text.substring(text.indexOf(character) - 18, text.indexOf(character) + 12)));
});
