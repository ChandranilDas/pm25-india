const fs = require('fs');
const path = require('path');

const outDir = path.join(__dirname, 'public');
fs.mkdirSync(outDir, { recursive: true });

// Copy index.html
fs.copyFileSync('index.html', path.join(outDir, 'index.html'));
console.log('Copied index.html to public/index.html');

// Copy Report.pdf and presentation if they exist
if (fs.existsSync('Report.pdf')) {
  fs.copyFileSync('Report.pdf', path.join(outDir, 'Report.pdf'));
}
if (fs.existsSync('presentation_5slides.pptx')) {
  fs.copyFileSync('presentation_5slides.pptx', path.join(outDir, 'presentation_5slides.pptx'));
}

// Copy results directory (excluding heavy training predictions / raw dumps)
const skipFiles = new Set(['random_forest.joblib', 'test_predictions.csv', 'complete_test_predictions.csv']);

function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      if (skipFiles.has(entry.name)) {
        console.log(`Skipping non-essential large file: ${entry.name}`);
        continue;
      }
      const stat = fs.statSync(srcPath);
      if (stat.size < 30 * 1024 * 1024) {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }
}

if (fs.existsSync('results')) {
  copyDir('results', path.join(outDir, 'results'));
  console.log('Copied results/ to public/results/');
}

console.log('Build completed successfully.');
