import {compile} from 'svelte/compiler';
import {build} from 'esbuild';
import {readFileSync} from 'node:fs';
await build({entryPoints:['main.js'],bundle:true,format:'iife',minify:true,outfile:'bundle.js',plugins:[{name:'svelte',setup(b){b.onLoad({filter:/\.svelte$/},args=>({contents:compile(readFileSync(args.path,'utf8'),{filename:args.path,generate:'client',css:'injected'}).js.code,loader:'js'}));}}]});
