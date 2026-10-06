/** Edit inspected template objects with Artifact Tool; never execute plan code. */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const [modulePath, skillDir, planPath, templatePath, registryPath, workDir, workspaceDir] = process.argv.slice(2);
if (![modulePath, skillDir, planPath, templatePath, registryPath, workDir, workspaceDir].every(value => value && path.isAbsolute(value))) {
  throw new Error('Renderer paths must be absolute.');
}
const {FileBlob, PresentationFile} = await import(pathToFileURL(modulePath).href);
process.env.RUNTIME_NODE_MODULES = path.resolve(path.dirname(modulePath), '../../..');
const runtimeRequire = createRequire(path.join(process.env.RUNTIME_NODE_MODULES, '__renderer__.cjs'));
const {createCanvas} = runtimeRequire('@napi-rs/canvas');
const measure = createCanvas(1,1).getContext('2d');
const {finalizePresentation} = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const plan = JSON.parse(await fs.readFile(planPath, 'utf8'));
const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
const hash = crypto.createHash('sha256').update(await fs.readFile(templatePath)).digest('hex');
if (hash !== registry.sha256 || registry.id !== plan.template_id) throw new Error('Template registry does not match the deck.');
if (plan.slides.length !== 10) throw new Error('Exactly 10 slides are required.');
const presentation = await PresentationFile.importPptx(await FileBlob.load(templatePath));
const originals = [...presentation.slides.items];
const updated = [];
const generated = [];

function find(slide, name) {
  const shape = slide.shapes.items.find(item => item.name === name);
  if (!shape) throw new Error(`Missing inspected template shape: ${name}`);
  return shape;
}
function setText(slide, entry, id, text, frame, size=24, font='Verdana', color='#F2F2F2') {
  const shape = find(slide, entry.shape_names[id]);
  shape.text = text;
  if (frame) shape.position = {left:frame[0], top:frame[1], width:frame[2], height:frame[3]};
  shape.text.style = {typeface:font, fontSize:size, color, autoFit:'none', verticalAlignment:'top', alignment:'left', bold:false, italic:false};
  updated.push({slideNumber:generated.length, name:shape.name, size, font, frame});
}
const lines = values => values.map(value => `• ${value}`).join('\n\n');
for (const spec of plan.slides) {
  const layout = spec.number === 1 ? 'cover' : spec.code ? 'code' : spec.layout;
  const entry = registry.layouts[layout];
  if (!entry) throw new Error(`Unsupported layout: ${layout}`);
  const slide = originals[entry.source_slide - 1].duplicate();
  generated.push(slide);
  for (const name of entry.clear_text_shapes) find(slide, name).text = '';
  const bullets = spec.bullets;
  if (layout === 'cover') {
    const shape = find(slide, entry.shape_names['3']);
    shape.text = [
      ...(plan.day == null ? [] : [{runs:[{run:`DAY ${plan.day}`, textStyle:{fontSize:'20px',bold:true,color:'#00E5FF',typeface:'Verdana'}}]}]),
      {runs:[{run:spec.title, textStyle:{fontSize:'46px',bold:true,color:'#F2F2F2',typeface:'Trebuchet MS'}}]},
      {runs:[{run:spec.title.trim().toLowerCase()===plan.topic.trim().toLowerCase() ? '' : plan.topic, textStyle:{fontSize:'24px',color:'#F2F2F2',typeface:'Verdana'}}]},
    ];
    shape.text.style = {autoFit:'none', verticalAlignment:'top'};
    updated.push({slideNumber:generated.length,name:shape.name,size:46,font:'Trebuchet MS',cover:true,frame:[168.96,172.8,801.6,233.19]});
    ['5','7','9'].forEach((id,index) => setText(slide,entry,id,
      plan.slides[index+1].title, [Number(id)==5?110.4:Number(id)==7?484.8:859.2, 477,316.8,105],22));
  } else {
    setText(slide,entry,'3',plan.day == null ? plan.topic : `DAY ${plan.day} | ${plan.topic}`, [76.8,38.4,1123.2,28],14,'Verdana','#00E5FF');
    setText(slide,entry,'4',`${spec.number-1}. ${spec.title}`, [76.8,80,1123.2,62],32,'Trebuchet MS');
    if (layout === 'code') {
      setText(slide,entry,'6','CODE EXAMPLE',[96,183,500,40],20,'Verdana','#00E5FF');
      setText(slide,entry,'7',spec.code,[96,233,512,390],20,'Consolas');
      setText(slide,entry,'9',lines(bullets),[682,181,499,436],24);
    } else if (layout === 'concept') {
      const split = Math.ceil(bullets.length/2);
      setText(slide,entry,'6','KEY POINTS',[106,180,480,40],20,'Verdana','#00E5FF');
      setText(slide,entry,'7',lines(bullets.slice(0,split)),[100,233,500,380],24);
      setText(slide,entry,'9',lines(bullets.slice(split)),[682,176,499,185],24);
      const refs = spec.source_ids.map(id => {
        const source=plan.sources.find(value=>value.source_id===id);
        return source.kind==='document' ? `${source.filename}, page ${source.page}` : new URL(source.url).hostname;
      });
      setText(slide,entry,'11','SOURCES\n'+[...new Set(refs)].join('\n'),[682,424,499,166],20,'Verdana','#FFB000');
    } else if (layout === 'comparison') {
      const split=Math.ceil(bullets.length/2);
      setText(slide,entry,'6',lines(bullets.slice(0,split)),[96,182,508.8,288],24);
      setText(slide,entry,'8',lines(bullets.slice(split)),[672,182,508.8,288],24);
      setText(slide,entry,'10',plan.topic,[96,543,1084.8,90],24,'Verdana','#00E5FF');
    } else if (layout === 'application') {
      ['6','8','10','12'].forEach((id,index)=>setText(slide,entry,id,bullets[index]??'',
        [91.2+index*283.2,180,230.4,254],22));
      setText(slide,entry,'14',bullets.slice(4).join('\n'),[96,506,1084.8,124],24);
    } else if (layout === 'challenge') {
      setText(slide,entry,'6','CHALLENGE',[96,184,480,40],20,'Verdana','#00E5FF');
      setText(slide,entry,'7',bullets[0]??'',[96,238,480,370],24);
      ['9','11','13','15'].forEach((id,index)=>setText(slide,entry,id,bullets[index+1]??'',
        [644,171+index*124.8,536,72],22));
    } else if (layout === 'takeaways') {
      ['6','8','10'].forEach((id,index)=>setText(slide,entry,id,bullets[index]??'',
        [96+index*374.4,184,302,252],24));
      setText(slide,entry,'12',lines(bullets.slice(3)),[96,504,1084.8,124],24);
    }
  }
  // All evidence and generated speaker notes stay in editable native notes.
  const references = spec.source_ids.map(id => {
    const source=plan.sources.find(value=>value.source_id===id);
    if (!source) throw new Error('Unknown source reference.');
    return `${id}: ${source.kind==='document' ? `${source.filename}, page ${source.page}` : source.url}\nEvidence: ${source.excerpt}`;
  });
  slide.speakerNotes.textFrame.setText([spec.speaker_notes??'',...references].filter(Boolean).join('\n\n'));
}
// Remove unused template panels rather than leaving empty outlined boxes.
const optional = {
 application:[[5,6],[7,8],[9,10],[11,12],[13,14]],
 challenge:[[8,9],[10,11],[12,13],[14,15]],
 takeaways:[[5,6],[7,8],[9,10],[11,12]],
};
for (let index=0; index<generated.length; index++) {
 const spec=plan.slides[index];
 const layout=spec.number===1 ? 'cover' : spec.code ? 'code' : spec.layout;
 const entry=registry.layouts[layout];
 for (const [panelId,textId] of optional[layout]??[]) {
  const textShape=find(generated[index],entry.shape_names[textId]);
  if (!String(textShape.text).trim()) {
   generated[index].shapes.deleteById(find(generated[index],entry.shape_names[panelId]).id);
   generated[index].shapes.deleteById(textShape.id);
  }
 }
}
for (const slide of originals) slide.delete();
generated.forEach((slide,index)=>slide.moveTo(index));
if (presentation.slides.items.length!==10) throw new Error('Rendered slide count mismatch.');
await fs.mkdir(workDir,{recursive:true});
for (let index=0;index<presentation.slides.items.length;index++) {
  const slide=presentation.slides.items[index];
  const layout=JSON.parse(await (await slide.export({format:'layout'})).text());
  await fs.writeFile(path.join(workDir,`slide-${index+1}.layout.json`),JSON.stringify(layout,null,2));
  for (const target of updated.filter(value=>value.slideNumber===index+1)) {
    const element=layout.elements.find(value=>value.name===target.name);
    if (!element?.text) continue;
    const insets=element.resolvedTextStyle?.insets??{left:9.6,right:9.6,top:4.8,bottom:4.8};
    const frame=target.frame??element.bbox;
    const width=frame[2]-insets.left-insets.right;
    let height=0;
    for (const [lineIndex,line] of element.text.split('\n').entries()) {
      const sizes=plan.day == null ? [46,24] : [20,46,24];
      const size=target.cover ? sizes[Math.min(lineIndex,sizes.length-1)] : target.size;
      measure.font=`${size}px "${target.font}"`;
      let current='', count=1;
      for (const word of line.split(/\s+/)) {
        if (measure.measureText(word).width>width) throw new Error(`Slide ${index+1}: an unbroken word or code token exceeds its text frame.`);
        const next=current ? `${current} ${word}` : word;
        if (measure.measureText(next).width>width) {count++;current=word;}
        else current=next;
      }
      height+=count*size*1.2;
    }
    // Measure wrapped text with template fonts; layout lineCount counts hard breaks only.
    if (height>frame[3]-insets.top-insets.bottom) {
      throw new Error(`Slide ${index+1}: content does not fit ${target.name}; shorten the text.`);
    }
  }
  await fs.writeFile(path.join(workDir,`slide-${index+1}.png`),new Uint8Array(await (await slide.export({format:'png',scale:1})).arrayBuffer()));
}
const candidatePath=path.join(workDir,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
await fs.mkdir(path.join(workDir,'final'),{recursive:true});
await finalizePresentation({
  workspaceDir,candidatePath,finalPath:path.join(workDir,'final','validated.pptx'),
  explicitTotalSlideCount:10,
  pythonExecutable:process.env.RENDER_VALIDATION_PYTHON,
  integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu',registry.slide_size_emu.join(','),'--validate-heading-fit'],
  fontPolicy:{basis:'reference',families:registry.fonts,referencePath:templatePath,referenceSha256:hash},
  requiredNativeTableOwnerSlides:[],
  verifyArtifactToolImport:true,
  receiptPath:path.join(workDir,'validation.json'),
});
console.log('Rendered and validated 10 editable slides.');
