/* Duration of each saved ground-truth video, read from the MP4's own metadata by the browser. */
(() => {
  const cache = new Map();
  function probe(url) {
    if (cache.has(url)) return cache.get(url);
    const promise = new Promise(resolve => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      video.muted = true;
      const timer = setTimeout(() => done(null), 15000);
      function done(value) {
        clearTimeout(timer);
        video.onloadedmetadata = video.onerror = null;
        video.removeAttribute('src');
        video.load();
        if (value === null) cache.delete(url);
        resolve(value);
      }
      video.onloadedmetadata = () => done(Number.isFinite(video.duration) ? video.duration : null);
      video.onerror = () => done(null);
      video.src = url;
    });
    cache.set(url, promise);
    return promise;
  }
  window.formatVideoTime = seconds => {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '—';
    const tenths = Math.round(Math.max(0, seconds) * 10);
    const minutes = Math.floor(tenths / 600), rest = (tenths % 600) / 10;
    return minutes ? `${minutes}m ${rest.toFixed(1).padStart(4, '0')}s` : `${rest.toFixed(1)}s`;
  };
  window.groundTruthVideoDurations = async (root, runs) => {
    const ready = runs.filter(run => run.artifact_status === 'ready');
    const durations = new Map();
    let cursor = 0;
    await Promise.all(Array.from({length: 4}, async () => {
      while (cursor < ready.length) {
        const run = ready[cursor++];
        const version = (run.video_sha256 || run.id).slice(0, 12);
        const seconds = await probe(`${root}${run.variant.split('-')[0]}/${run.variant}/video.mp4?v=${version}`);
        if (seconds !== null) durations.set(run.variant, seconds);
      }
    }));
    return durations;
  };
})();
