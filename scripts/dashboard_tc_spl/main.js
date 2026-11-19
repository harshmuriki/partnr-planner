import {mount} from 'svelte';
import App from './App.svelte';
// Paper order (ReAct, VLM-TAMP, TRU-POMDP); colour/line style/marker are tied to the baseline, not its position.
const STYLE={react_images:{color:'#D55E00',dash:'9 5',style:1},vlm_tamp_images:{color:'#0072B2',dash:'',style:0},tru_pomdp_no_images:{color:'#CC79A7',dash:'2 5',style:2}};
const ORDER=['react_images','vlm_tamp_images','tru_pomdp_no_images'];
const data=JSON.parse(document.getElementById('tc-spl-data').textContent);
data.baselines=[...data.baselines].sort((a,b)=>ORDER.indexOf(a.key)-ORDER.indexOf(b.key)).map(b=>({...b,...STYLE[b.key]}));
mount(App,{target:document.getElementById('tc-spl-root'),props:{data}});
