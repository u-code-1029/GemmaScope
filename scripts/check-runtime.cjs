const fs = require('node:fs');
const path = require('node:path');
module.exports = async function beforePack(context) {
  const runtime = path.join(context.packager.projectDir, 'runtime', 'python', context.electronPlatformName === 'win32' ? 'python.exe' : 'bin/python3');
  if (!fs.existsSync(runtime)) throw new Error('Packaged apps require a standalone Python runtime. Run npm run runtime:prepare first. See README.md.');
};
