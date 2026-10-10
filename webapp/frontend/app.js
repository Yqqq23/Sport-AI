const $ = id => document.getElementById(id);
const labels = {queued: '排队等待', processing: '正在分析', encoding: '正在转码为网页视频', completed: '分析完成', failed: '分析失败'};
let jobId = null, timer = null, selectedNumber = null, rosterKey = '', uploading = false;
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请检查服务器日志');
  return data;
}
function setVideo(id, url) {
  if ($(id).getAttribute('src') !== url) { $(id).src = url; $(id).load(); }
  $(id + '-download').href = url;
}
function render(job) {
  $('task-panel').hidden = false;
  $('task-name').textContent = job.filename;
  $('task-status').textContent = `${labels[job.status]} · ${job.progress}%`;
  $('progress').value = job.progress;
  $('log-link').href = job.log_url;
  $('task-detail').textContent = `任务 ${job.id} · 分析上限 ${job.total_frames} 帧。处理耗时取决于视频长度与服务器 GPU。`;
  $('upload-button').disabled = uploading || !['completed', 'failed'].includes(job.status);
  $('results').hidden = job.status !== 'completed';
  if (job.status === 'failed') showError(job.error);
  if (job.status !== 'completed') return;
  for (const kind of ['topdown', 'annotated']) setVideo(kind, job.videos[kind]);
  const key = JSON.stringify(job.players);
  if (rosterKey !== key) {
    rosterKey = key;
    $('player-select').replaceChildren(...job.players.map(p => new Option(`#${p.number} · ${p.frames} 帧`, p.number)));
  }
  const state = selectedNumber ? job.selections[selectedNumber] : null;
  $('player-button').disabled = !job.players.length || state?.status === 'processing';
  $('player-select').disabled = !job.players.length;
  $('selected-result').hidden = state?.status !== 'completed';
  $('player-status').textContent = !job.players.length ? '没有识别到稳定的球衣号码。请查看日志，检查识别模型和视频清晰度。' :
    state?.status === 'processing' ? `正在排队或生成 #${selectedNumber} 球员视频…` :
    state?.status === 'failed' ? `生成失败：${state.error}，可以重新点击生成。` :
    state?.status === 'completed' ? `#${selectedNumber} 视频已就绪` : '请选择号码并生成视频。';
  if (job.calibrated === false) $('player-status').textContent += ' 当前未配置标定，俯视路线不可用。';
  if (state?.status === 'completed') {
    $('selected-title').textContent = `#${selectedNumber} · 高亮画面 + 累计运动路线`;
    setVideo('selected', state.url);
  }
}
async function poll() {
  clearTimeout(timer);
  const current = jobId;
  if (!current) return;
  try {
    const job = await api(`/api/jobs/${current}`);
    if (current !== jobId) return;
    showError(''); render(job);
    if (['queued', 'processing', 'encoding'].includes(job.status) || Object.values(job.selections).some(s => s.status === 'processing')) timer = setTimeout(poll, 2000);
  } catch (error) {
    if (current !== jobId) return;
    showError(`${error.message}。可重新上传；5 秒后重试查询。`);
    $('upload-button').disabled = false;
    timer = setTimeout(poll, 5000);
  }
}
$('upload-form').addEventListener('submit', event => {
  event.preventDefault();
  const file = $('video-file').files[0];
  if (!file) return;
  clearTimeout(timer); jobId = null; uploading = true; selectedNumber = null; rosterKey = '';
  localStorage.removeItem('basketball-job');
  $('results').hidden = true; $('task-panel').hidden = true; showError('');
  $('upload-button').disabled = true;
  const body = new FormData(); body.append('video', file);
  const xhr = new XMLHttpRequest(); xhr.open('POST', '/api/jobs');
  xhr.upload.onprogress = event => { if (event.lengthComputable) $('upload-button').textContent = `上传中 ${Math.round(100 * event.loaded / event.total)}%`; };
  xhr.onload = () => {
    uploading = false; $('upload-button').textContent = '开始分析 ↗';
    try {
      const result = JSON.parse(xhr.responseText);
      if (xhr.status >= 400) throw new Error(result.detail || '上传失败');
      jobId = result.id; localStorage.setItem('basketball-job', jobId); render(result); poll();
    } catch (error) { showError(error.message); $('upload-button').disabled = false; }
  };
  xhr.onerror = () => { uploading = false; showError('上传连接失败，请检查网络后重试'); $('upload-button').disabled = false; $('upload-button').textContent = '开始分析 ↗'; };
  xhr.send(body);
});
$('player-select').addEventListener('change', () => {
  selectedNumber = null; $('selected-result').hidden = true;
  $('player-status').textContent = '点击生成，查看当前号码。'; $('player-button').disabled = false;
});
$('player-form').addEventListener('submit', async event => {
  event.preventDefault(); selectedNumber = $('player-select').value;
  $('player-button').disabled = true; showError('');
  try { render(await api(`/api/jobs/${jobId}/players/${encodeURIComponent(selectedNumber)}`, {method: 'POST'})); poll(); }
  catch (error) { showError(error.message); $('player-button').disabled = false; }
});
api('/api/health').then(health => {
  $('limits').textContent = `处理前 ${health.max_frames} 帧 · Per-digit 球衣识别 · 最大 ${health.max_upload_mb} MB`;
  if (!health.ffmpeg) showError('服务器缺少 FFmpeg，请安装后再上传视频。');
}).catch(error => showError(error.message));
jobId = localStorage.getItem('basketball-job');
if (jobId) poll();
