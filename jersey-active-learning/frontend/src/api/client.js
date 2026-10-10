export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

function buildUrl(path) {
  return `${API_BASE_URL}${path}`
}

async function request(path, options = {}) {
  const url = buildUrl(path)
  try {
    const response = await fetch(url, options)
    if (!response.ok) {
      const text = await response.text()
      const error = new Error(`Request failed: ${response.status} ${response.statusText}`)
      error.requestUrl = url
      error.status = response.status
      error.responseText = text
      throw error
    }
    return response.json()
  } catch (error) {
    if (!error.requestUrl) {
      error.requestUrl = url
    }
    throw error
  }
}

export async function fetchVideos() {
  return request('/videos')
}

export async function uploadVideo(file) {
  const body = new FormData()
  body.append('file', file)
  return request('/upload-video', {
    method: 'POST',
    body,
  })
}

export async function fetchAnnotationCandidates(videoId, threshold = 0.7) {
  const url = `${API_BASE_URL}/videos/${videoId}/annotation-candidates?threshold=${threshold}`
  console.log('fetchAnnotationCandidates URL', url)
  const response = await fetch(url)
  const text = await response.text()
  console.log('fetchAnnotationCandidates response status', response.status)
  console.log('fetchAnnotationCandidates raw text', text)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${text}`)
  }
  const json = JSON.parse(text)
  console.log('fetchAnnotationCandidates parsed json', json)
  return json
}

export async function fetchReviewItems(videoId) {
  return request(`/videos/${videoId}/review-items`)
}

export async function fetchAnnotations(videoId) {
  return request(`/videos/${videoId}/annotations`)
}

export async function fetchAnnotationProgress(videoId) {
  return request(`/videos/${videoId}/annotation-progress`)
}

export async function fetchRecognitionSummary(videoId) {
  return request(`/videos/${videoId}/recognition-summary`)
}

export async function exportInternalDataset(videoId) {
  return request(`/videos/${videoId}/export-internal-dataset`, {
    method: 'POST',
  })
}

export async function fetchVideoExports(videoId) {
  return request(`/videos/${videoId}/exports`)
}

export async function exportCocoDataset(videoId) {
  return request(`/videos/${videoId}/export-coco-dataset`, {
    method: 'POST',
  })
}

export async function fetchCocoExports(videoId) {
  return request(`/videos/${videoId}/coco-exports`)
}

export async function downloadExport(exportId) {
  return request(`/exports/${exportId}/download`)
}

export async function downloadCocoExport(exportId) {
  return request(`/coco-exports/${exportId}/download`)
}

export async function exportYoloDataset(videoId) {
  return request(`/videos/${videoId}/export-yolo-dataset`, {
    method: 'POST',
  })
}

export async function fetchYoloExports(videoId) {
  return request(`/videos/${videoId}/yolo-exports`)
}

export async function downloadYoloExport(exportId) {
  return request(`/yolo-exports/${exportId}/download`)
}

export async function createDatasetVersion(payload) {
  return request('/dataset-versions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export async function fetchDatasetVersions() {
  return request('/dataset-versions')
}

export async function fetchDatasetVersion(datasetVersionId) {
  return request(`/dataset-versions/${datasetVersionId}`)
}

export async function fetchDatasetVersionManifest(datasetVersionId) {
  return request(`/dataset-versions/${datasetVersionId}/manifest`)
}

export async function fetchDatasetVersionSplitReport(datasetVersionId) {
  return request(`/dataset-versions/${datasetVersionId}/split-report`)
}

export async function downloadDatasetVersion(datasetVersionId) {
  return request(`/dataset-versions/${datasetVersionId}/download`)
}

export async function rebuildDatasetVersion(datasetVersionId) {
  return request(`/dataset-versions/${datasetVersionId}/rebuild`, {
    method: 'POST',
  })
}

export async function saveAnnotation(payload) {
  return request('/annotations', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export async function updateAnnotation(annotationId, payload) {
  return request(`/annotations/${annotationId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export async function deleteAnnotation(annotationId) {
  return request(`/annotations/${annotationId}`, {
    method: 'DELETE',
  })
}

export async function runRecognition(videoId) {
  return request(`/videos/${videoId}/run-jersey-recognition`, {
    method: 'POST',
  })
}

export async function runRecognitionAsync(videoId) {
  return request(`/videos/${videoId}/run-jersey-recognition-async`, {
    method: 'POST',
  })
}

export async function runYoloPreannotationAsync(videoId) {
  return request(`/videos/${videoId}/run-yolo-preannotation-async`, {
    method: 'POST',
  })
}

export async function fetchJob(jobId) {
  return request(`/jobs/${jobId}`)
}

export async function pollJob(jobId, { interval = 1000, onUpdate, shouldStop } = {}) {
  return new Promise((resolve, reject) => {
    let stopped = false

    const tick = async () => {
      if (stopped) return
      if (shouldStop?.()) {
        stopped = true
        resolve(null)
        return
      }
      try {
        const job = await fetchJob(jobId)
        console.log('Job poll response:', job)
        if (shouldStop?.()) {
          stopped = true
          resolve(null)
          return
        }
        if (onUpdate) onUpdate(job)
        if (job.status === 'completed' || job.status === 'failed') {
          stopped = true
          resolve(job)
          return
        }
        setTimeout(() => {
          if (shouldStop?.()) {
            stopped = true
            resolve(null)
            return
          }
          tick()
        }, interval)
      } catch (error) {
        stopped = true
        reject(error)
      }
    }

    tick()
  })
}
