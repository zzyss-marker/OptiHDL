#!/usr/bin/env node
/**
 * Bridge script: Verilog → DigitalJS JSON
 *
 * Reads JSON from stdin: { "code": "module ...", "module": "top" }
 * Writes JSON to stdout: { "success": true, "output": { ... } }
 *
 * Internally calls yosys2digitaljs.process_files() which invokes Yosys.
 */
const y2d = require('yosys2digitaljs');

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', async () => {
    try {
        const data = JSON.parse(input);
        const code = data.code || '';
        const mod  = data.module || 'top';

        const result = await y2d.process_files(
            { 'input.v': code },
            { optimize: false }
        );

        const out = JSON.stringify({ success: true, output: result.output });
        process.stdout.write(out);
    } catch (e) {
        const out = JSON.stringify({ success: false, error: String(e.message || e) });
        process.stdout.write(out);
        process.exitCode = 0;  // don't crash — let Python read the JSON error
    }
});
